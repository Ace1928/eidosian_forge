import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.distribute import distributed_training_utils_v1 as dist_utils
from tensorflow.python.keras.engine import partial_batch_padding_handler as padding_util
from tensorflow.python.keras.engine import training_arrays_v1
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import tf_logging as logging
def experimental_tpu_predict_loop(model, dataset, verbose=0, steps=None, callbacks=None):
    """Predict loop for predicting with TPU tf.distribute.Strategy.

  Args:
      model: Keras Model instance.
      dataset: Dataset for input data.
      verbose: Integer, Verbosity mode 0 or 1.
      steps: Total number of steps (batches of samples)
          before declaring `_predict_loop` finished.
          Ignored with the default value of `None`.
      callbacks: List of callbacks to be called during training

  Returns:
      Array of predictions (if the model has a single output)
      or list of arrays of predictions
      (if the model has multiple outputs).
  """
    mode = ModeKeys.PREDICT
    dataset_fully_shaped = dist_utils.is_dataset_shape_fully_defined(dataset)
    padding_handler = None
    if not dataset_fully_shaped:
        padding_handler = padding_util.PartialBatchPaddingHandler(model._feed_output_shapes)
        batch_size, _, prefetch_buffer = input_lib._get_dataset_attributes(dataset)
        padding_handler.padded_batch_size = batch_size
        padding_handler.padding_mask = dataset.reduce(padding_handler.padding_mask, padding_handler.update_mask)
        dataset = dataset.map(padding_handler.pad_batch)
        dataset = dataset.unbatch()
        dataset = dataset.batch(batch_size, drop_remainder=True)
        if prefetch_buffer is not None:
            dataset = dataset.prefetch(prefetch_buffer)
    current_strategy = model._distribution_strategy
    iterator = dist_utils.get_iterator(dataset, current_strategy)
    scope = dist_utils.distributed_scope(strategy=current_strategy, learning_phase=0)
    scope.__enter__()

    def _predict_step_fn(inputs):
        """A fn that returns output of single prediction step."""
        distribute_lib.get_replica_context().merge_call(_build_model, args=(model, mode, inputs))
        _, outputs, updates, _ = _per_replica_execution_function(dist_utils.get_distributed_model(model, mode), mode)
        with ops.control_dependencies([updates]):
            return [array_ops.identity(out) for out in outputs]
    predict_input_data = iterator.get_next()
    per_replica_outputs = current_strategy.run(_predict_step_fn, args=(predict_input_data,))
    output_tensors = dist_utils.flatten_per_replica_values(current_strategy, per_replica_outputs)
    if verbose >= 1:
        progbar = Progbar(target=steps)
    if model._compile_distribution:
        dist_utils._copy_weights_to_distributed_model(model, mode)
    dist_utils._reset_metrics(model)
    callbacks = cbks.configure_callbacks(callbacks, model, do_validation=False, epochs=1, steps_per_epoch=steps, verbose=verbose, count_mode='steps', mode=mode)
    callbacks._call_begin_hook(mode)
    num_model_outputs = len(model.output_names)
    unconcatenated_outs = [[] for _ in range(num_model_outputs)]
    if steps is not None:
        target_steps = steps
    else:
        raise ValueError('Number of steps could not be inferred from the data, please pass the steps argument.')
    current_step = 0
    while current_step < target_steps:
        batch_logs = {'batch': current_step, 'size': 1}
        callbacks._call_batch_hook(mode, 'begin', current_step, batch_logs)
        try:
            predict_ops = control_flow_ops.group(output_tensors)
            _, batch_outs = backend.batch_get_value([predict_ops, output_tensors])
        except errors.OutOfRangeError:
            warning_msg = 'Make sure that your dataset can generate at least `steps` batches (in this case, {} batches).'.format(steps)
            logging.warning('Your dataset iterator ran out of data; interrupting evaluation. ' + warning_msg)
            break
        for i in range(num_model_outputs):
            output_start_index = i * current_strategy.num_replicas_in_sync
            output_end_index = output_start_index + current_strategy.num_replicas_in_sync
            single_model_output = batch_outs[output_start_index:output_end_index]
            unconcatenated_outs[i].extend(single_model_output)
        batch_logs = cbks.make_logs(model, batch_logs, batch_outs, mode)
        callbacks._call_batch_hook(mode, 'end', current_step, batch_logs)
        if verbose == 1:
            progbar.update(current_step + 1)
        current_step += 1
    if verbose >= 1:
        progbar.update(current_step)
    callbacks._call_end_hook(mode)
    scope.__exit__(None, None, None)
    if len(unconcatenated_outs) == 1:
        prediction_result = np.concatenate(unconcatenated_outs[0], axis=0)
    else:
        prediction_result = [np.concatenate(out, axis=0) for out in unconcatenated_outs]
    if padding_handler:
        prediction_result = padding_handler.apply_mask(prediction_result)
    return prediction_result
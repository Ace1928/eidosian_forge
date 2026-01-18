import collections
import enum
from typing import Any, Callable, Iterable, List, Optional, Text, Tuple, Union
from absl import logging
import numpy as np
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import dynamic_padding_pb2 as dynamic_padding
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as embedding_pb2
from tensorflow.python import tf2
from tensorflow.python.compiler.xla import xla
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_name_util
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def split_compile_and_replicate(computation: Callable[..., Any], inputs: Optional[List[List[core_types.Tensor]]]=None, infeed_queue: Optional[tpu_feed.InfeedQueue]=None, device_assignment: Optional[device_assignment_lib.DeviceAssignment]=None, name: Optional[Text]=None, use_tpu: bool=True, maximum_shapes: Optional[Any]=None, padding_spec: Optional[PaddingSpec]=None, xla_options: Optional[XLAOptions]=None) -> List[List[core_types.Tensor]]:
    """Builds graph operators that runs compilation and replicated computation.

  This is a lower level interface than replicate that returns a separate compile
  and execute output tensor. In the generated graph the compile op feeds into
  the execute op and no additional compilation is incurred when running the
  compile op before the execute op. The compile op returns additional
  information about the compilation but does not return the compiled program.

  Args:
    computation: A Python function that builds the computation to replicate.
    inputs: A list of lists of input tensors or `None` (equivalent to
      `[[]]`), indexed by `[replica_num][input_num]`. All replicas must
      have the same number of inputs. Each input can be a nested structure
      containing values that are convertible to tensors. Note that passing an
      N-dimension list of compatible values will result in a N-dimension list of
      scalar tensors rather than a single Rank-N tensors. If you need different
      behavior, convert part of inputs to tensors with `tf.convert_to_tensor`.
    infeed_queue: If not `None`, the `InfeedQueue` from which to append a tuple
      of arguments as inputs to computation.
    device_assignment: If not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. Uses a default device assignment if `None`. The
      `DeviceAssignment` may be omitted if each replica of the computation uses
      only one core, and there is either only one replica, or the number of
      replicas is equal to the number of cores in the TPU system.
    name: (Deprecated) Does nothing.
    use_tpu: When false, the input `computation` is executed on the XLA CPU/GPU
      backends. Currently, only supports a default placement (computation is
      placed on GPU if one is available, and on CPU if not).
    maximum_shapes: A nested structure of tf.TensorShape representing the shape
      to which the respective component of each input element in each replica
      should be padded. Any unknown dimensions (e.g.
      tf.compat.v1.Dimension(None) in a tf.TensorShape or -1 in a tensor-like
      object) will be padded to the maximum size of that dimension over all
      replicas. The structure of `maximum_shapes` needs to be the same as
      `inputs[0]`.
    padding_spec: An enum specified by `tf.tpu.PaddingSpec`. This describes the
      padding policy when the `inputs` to `tf.tpu.replicate` is dynamic.
      One usage is to enable automatic bucketizing on the inputs by setting the
      value to `tpu.PaddingSpec.POWER_OF_TWO`, which can help to reduce the
      recompilation in the XLA side.
    xla_options: An instance of `tpu.XLAOptions` which indicates the options
      passed to XLA compiler. Use `None` for default options.

  Returns:
    A list of lists with the first list corresponding to the compile op and the
    second a list of output tensors, indexed by `[replica_num][output_num]`.
  Raises:
    ValueError: If all replicas do not have equal numbers of input tensors.
    ValueError: If the number of inputs per replica does not match
      the number of formal parameters to `computation`.
    ValueError: If the static `inputs` dimensions don't match with the values
      given in `maximum_shapes`.
    ValueError: If the structure of inputs per replica does not match
      the structure of `maximum_shapes`.
  """
    del name
    inputs = [[]] if inputs is None else inputs
    xla_options = xla_options or XLAOptions()
    metadata_kwargs = {}
    if device_assignment is not None:
        metadata_kwargs = {'topology': device_assignment.topology.serialized(), 'device_assignment': device_assignment.core_assignment.flatten().tolist()}
        metadata_kwargs['num_cores_per_replica'] = device_assignment.num_cores_per_replica
    metadata_kwargs['allow_soft_placement'] = config.get_soft_device_placement()
    if config.get_soft_device_placement():
        logging.info('Automatic outside compilation is enabled. Ops without XLA kernels will be automatically placed on CPU.')
    if not isinstance(inputs, list):
        raise TypeError(f'tpu.replicate() inputs must be a list of lists/tuples, received {type(inputs)}')
    if any((not isinstance(inp, (list, tuple)) for inp in inputs)):
        raise TypeError(f'tpu.replicate() inputs must be a list of lists/tuples, received types: {[type(inp) for inp in inputs]}')
    num_replicas = len(inputs)
    if num_replicas == 0:
        return []
    for i in range(1, num_replicas):
        nest.assert_same_structure(inputs[0], inputs[i])
    inputs = variable_utils.convert_variables_to_tensors(inputs)
    flat_inputs_with_nones = [nest.flatten(per_replica_input, expand_composites=True) for per_replica_input in inputs]
    is_composite = nest.flatten(nest.map_structure(lambda x: _flatten_and_filter_composite(x, False, True), inputs[0]))
    flat_inputs = []
    for inp in flat_inputs_with_nones:
        flat_inputs.append([constant_op.constant(0) if x is None else ops.convert_to_tensor(x) for x in inp])
    flat_input_types = [x.dtype for x in flat_inputs[0]]
    input_arity = len(inputs[0])
    flat_input_arity = len(flat_input_types)
    for i in range(num_replicas):
        if len(inputs[i]) != input_arity:
            raise ValueError('Replicas must have the same number of inputs. Replica 0 had {} inputs, replica {} had {} inputs.'.format(input_arity, i, len(inputs[i])))
        types = [x.dtype for x in flat_inputs[i]]
        if types != flat_input_types:
            raise ValueError('Replicas must have matching input types. Replica 0 had input types {}, replica {} had input types {}'.format(flat_input_types, i, types))
    arg_error = xla.check_function_argument_count(computation, input_arity, infeed_queue)
    if arg_error is not None:
        if infeed_queue is None:
            raise TypeError(f'Supplied computation cannot be called with the specified inputs. You specified {input_arity} inputs: {[i.name for i in inputs[0]]}, but the computation needs {arg_error}')
        else:
            raise TypeError(f'Supplied computation cannot be called with the specified inputs. You specified {input_arity} inputs: {[i.name for i in inputs[0]]} ', f'and {infeed_queue.number_of_tuple_elements} additional inputs from infeed, but the computation needs {arg_error}')
    dynamic_shape_inputs = False
    if maximum_shapes:
        if infeed_queue:
            raise ValueError('Dynamic input shapes are not supported with infeed queues')
        nest.assert_same_structure(inputs[0], maximum_shapes, check_types=False)
        flat_maximum_shapes = nest.flatten([_flatten_and_filter_composite(x, y) for x, y in zip(nest.flatten(inputs[0]), nest.flatten(maximum_shapes))])
        flat_maximum_shapes = [tensor_shape.TensorShape(s) if s is not None else None for s in flat_maximum_shapes]
        nest.assert_same_structure(flat_inputs[0], flat_maximum_shapes, check_types=False)
        unpadded_inputs = flat_inputs
        flat_inputs, padding_maps = _pad_all_input(unpadded_inputs, flat_maximum_shapes, padding_spec)
        if padding_maps:
            dynamic_shape_inputs = True
            logging.info('TPU has inputs with dynamic shapes: %s', inputs[0])
    metadata_kwargs['step_marker_location'] = getattr(computation, 'step_marker_location', 'STEP_MARK_AT_ENTRY')
    metadata_kwargs['use_spmd_for_xla_partitioning'] = xla_options.use_spmd_for_xla_partitioning
    graph = ops.get_default_graph()
    flat_replicated_inputs = []
    for i in range(0, len(flat_inputs[0])):
        replicas = [flat_inputs[replica][i] for replica in range(num_replicas)]
        flat_replicated_inputs.append(tpu_ops.tpu_replicated_input(replicas, name='input{}'.format(i)))
    if isinstance(graph, func_graph.FuncGraph):
        cluster_name = graph.unique_name('cluster_' + graph.name)
    else:
        cluster_name = graph.unique_name('cluster')
    pivot = control_flow_ops.no_op(name=cluster_name + '/pivot')
    pivot._set_attr(_PIVOT_FOR_CLUSTER, attr_value_pb2.AttrValue(s=compat.as_bytes(cluster_name)))
    context = tpu_replication.TPUReplicateContext(name=cluster_name, num_replicas=num_replicas, pivot=pivot)
    try:
        context.Enter()
        metadata = tpu_ops.tpu_replicate_metadata(num_replicas=num_replicas, use_tpu=use_tpu, **metadata_kwargs)
        with tpu_function.tpu_shard_context(num_replicas), ops.control_dependencies([metadata]):
            if dynamic_shape_inputs and xla_options.enable_xla_dynamic_padder:
                for padding_map in padding_maps:
                    input_shape = flat_replicated_inputs[padding_map.arg_index].shape
                    flat_replicated_inputs[padding_map.arg_index] = tf2xla.set_dynamic_dimension_size(flat_replicated_inputs[padding_map.arg_index], padding_map.shape_index, flat_replicated_inputs[padding_map.padding_arg_index])
                    flat_replicated_inputs[padding_map.arg_index].set_shape(input_shape)
            flat_replicated_inputs = [array_ops.identity(x, name='replicated_input_{}'.format(i)) for i, x in enumerate(flat_replicated_inputs)]
            for i, composite in zip(flat_replicated_inputs, is_composite):
                if not dynamic_shape_inputs or composite:
                    i.op._set_attr('_tpu_input_identity', attr_value_pb2.AttrValue(b=True))
            computation_inputs = [None if inp is None else replicated for replicated, inp in zip(flat_replicated_inputs, flat_inputs_with_nones[0])]
            computation_inputs = nest.pack_sequence_as(structure=inputs[0], flat_sequence=computation_inputs[:flat_input_arity], expand_composites=True)
            if infeed_queue is not None:
                infeed_queue.set_number_of_shards(num_replicas)
                for t in infeed_queue.generate_dequeue_op():
                    computation_inputs.append(t)
            vscope = variable_scope.get_variable_scope()
            saved_use_resource = vscope.use_resource
            saved_custom_getter = vscope.custom_getter

            def custom_getter(getter, name, *args, **kwargs):
                """Variables on TPU have a few restrictions."""
                partitioner = kwargs.get('partitioner', None)
                if partitioner is not None:
                    kwargs['partitioner'] = None
                    logging.warning('Partitioned variables are not supported on TPU. Got `partitioner` that is %s for variable %s. Setting `partitioner` to `None`.', partitioner, name)
                if saved_custom_getter is None:
                    return getter(name, *args, **kwargs)
                else:
                    return saved_custom_getter(getter, name, *args, **kwargs)
            vscope.set_use_resource(True)
            vscope.set_custom_getter(custom_getter)
            outputs = computation(*computation_inputs)
            vscope.set_use_resource(saved_use_resource)
            vscope.set_custom_getter(saved_custom_getter)
            outputs = variable_utils.convert_variables_to_tensors(outputs)
        need_spmd_partitioning = xla_options.use_spmd_for_xla_partitioning and device_assignment is not None and (device_assignment.num_cores_per_replica > 1)
        outputs_is_flat = xla.is_flat(outputs)
        if outputs_is_flat:
            output_tensors, control_deps, pack_template = _postprocess_flat_outputs(outputs, need_spmd_partitioning)
        else:
            output_tensors, control_deps, pack_template = _postprocess_non_flat_outputs(outputs, need_spmd_partitioning)
        if tensor_tracer.TensorTracer.is_enabled():
            if tf2.enabled():
                logging.warn('TF API ver >= 2.0 detected. Tensor Tracer v1 is not enabled.')
            else:
                tt = tensor_tracer.TensorTracer()
                output_tensors = tt.trace_tpu(ops.get_default_graph(), output_tensors, control_deps, num_replicas)
        context.ExitResult(output_tensors)
    finally:
        context.report_unsupported_operations()
        context.Exit()
        host_compute_core = context.HostComputeCore()
    if host_compute_core:
        attr_value = attr_value_pb2.AttrValue()
        attr_value.list.s.extend((compat.as_bytes(x) for x in host_compute_core))
        metadata._set_attr('host_compute_core', attr_value)
    with ops.control_dependencies([metadata]):
        if use_tpu:
            compile_status = tpu_ops.tpu_compilation_result()
            op = compile_status.op
            attr_value = attr_value_pb2.AttrValue(s=compat.as_bytes(cluster_name))
            op._set_attr(_TPU_COMPILATION_STATUS_ATTR, attr_value)
        else:
            compile_status = control_flow_ops.no_op(name='compilation_status')
    if not output_tensors:
        return [compile_status, [control_flow_ops.group(control_deps, name='shard_%d' % i) for i in range(num_replicas)]]
    replicated_outputs = [[] for i in range(num_replicas)]
    for i, t in enumerate(output_tensors):
        if t is None:
            for replica in range(num_replicas):
                replicated_outputs[replica].append(None)
            continue
        ys = tpu_ops.tpu_replicated_output(t, num_replicas, name='output{}'.format(i))
        with ops.control_dependencies(control_deps):
            for replica in range(num_replicas):
                replicated_outputs[replica].append(array_ops.identity(ys[replica], name='output_%d_shard_%d' % (i, replica)))
    replicated_outputs = [nest.pack_sequence_as(pack_template, replica_outs, expand_composites=True) for replica_outs in replicated_outputs]
    return [compile_status, replicated_outputs]
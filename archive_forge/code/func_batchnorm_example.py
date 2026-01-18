from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import step_fn
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.layers import core
from tensorflow.python.layers import normalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
def batchnorm_example(optimizer_fn, batch_per_epoch=1, momentum=0.9, renorm=False, update_ops_in_replica_mode=False):
    """Example of non-distribution-aware legacy code with batch normalization."""

    def dataset_fn():
        return dataset_ops.Dataset.from_tensor_slices([[[float(x * 8 + y + z * 100) for y in range(8)] for x in range(16)] for z in range(batch_per_epoch)]).repeat()
    optimizer = optimizer_fn()
    batchnorm = normalization.BatchNormalization(renorm=renorm, momentum=momentum, fused=False)
    layer = core.Dense(1, use_bias=False)

    def model_fn(x):
        """A model that uses batchnorm."""

        def loss_fn():
            y = batchnorm(x, training=True)
            with ops.control_dependencies(ops.get_collection(ops.GraphKeys.UPDATE_OPS) if update_ops_in_replica_mode else []):
                loss = math_ops.reduce_mean(math_ops.reduce_sum(layer(y)) - constant_op.constant(1.0))
            return loss
        if strategy_test_lib.is_optimizer_v2_instance(optimizer):
            return optimizer.minimize(loss_fn, lambda: layer.trainable_variables)
        return optimizer.minimize(loss_fn)
    return (model_fn, dataset_fn, batchnorm)
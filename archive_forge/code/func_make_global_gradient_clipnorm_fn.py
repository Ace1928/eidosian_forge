from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.ops import clip_ops
from tensorflow.python.platform import tf_logging as logging
def make_global_gradient_clipnorm_fn(clipnorm):
    """Creates a gradient transformation function for clipping by norm."""
    if clipnorm is None:
        return lambda grads_and_vars: grads_and_vars

    def gradient_clipnorm_fn(grads_and_vars):
        if isinstance(distribute_lib.get_strategy(), (central_storage_strategy.CentralStorageStrategy, central_storage_strategy.CentralStorageStrategyV1)):
            raise ValueError('`global_clipnorm` is not supported with `CenteralStorageStrategy`')
        grads, variables = zip(*grads_and_vars)
        clipped_grads, _ = clip_ops.clip_by_global_norm(grads, clipnorm)
        clipped_grads_and_vars = list(zip(clipped_grads, variables))
        return clipped_grads_and_vars
    return gradient_clipnorm_fn
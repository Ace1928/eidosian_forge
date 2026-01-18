from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer
from tensorflow.python.training import queue_runner
from tensorflow.python.training import session_manager
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def get_init_tokens_op(self, num_tokens=-1):
    """Returns the op to fill the sync_token_queue with the tokens.

    This is supposed to be executed in the beginning of the chief/sync thread
    so that even if the total_num_replicas is less than replicas_to_aggregate,
    the model can still proceed as the replicas can compute multiple steps per
    variable update. Make sure:
    `num_tokens >= replicas_to_aggregate - total_num_replicas`.

    Args:
      num_tokens: Number of tokens to add to the queue.

    Returns:
      An op for the chief/sync replica to fill the token queue.

    Raises:
      ValueError: If this is called before apply_gradients().
      ValueError: If num_tokens are smaller than replicas_to_aggregate -
        total_num_replicas.
    """
    if self._gradients_applied is False:
        raise ValueError('get_init_tokens_op() should be called after apply_gradients().')
    tokens_needed = self._replicas_to_aggregate - self._total_num_replicas
    if num_tokens == -1:
        num_tokens = self._replicas_to_aggregate
    elif num_tokens < tokens_needed:
        raise ValueError('Too few tokens to finish the first step: %d (given) vs %d (needed)' % (num_tokens, tokens_needed))
    if num_tokens > 0:
        with ops.device(self._global_step.device), ops.name_scope(''):
            tokens = array_ops.fill([num_tokens], self._global_step)
            init_tokens = self._sync_token_queue.enqueue_many((tokens,))
    else:
        init_tokens = control_flow_ops.no_op(name='no_init_tokens')
    return init_tokens
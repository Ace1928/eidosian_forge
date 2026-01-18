from tensorflow.python.eager import context
from tensorflow.python.eager import execute
def should_invoke_op_callbacks():
    """Determine if op callbacks are present and should be invoked.

  Returns:
    A thread-local result (boolean) indicating whether any op callback(s) exist
    and should be invoked.
  """
    ctx = context.context()
    return ctx.op_callbacks and (not ctx.invoking_op_callbacks)
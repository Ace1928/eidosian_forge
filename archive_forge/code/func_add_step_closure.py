import os
import threading
from queue import Empty as EmptyQueue, Queue
from torch._lazy.device_context import get_device_context
def add_step_closure(closure, args=(), run_async=False):
    """Adds a closure to the list of the ones to be run at the end of the step.
    Many times during model training there is the need to print/report (print to
    console, post to tensorboard, etc...) information which require the content of
    intermediary tensors to be inspected.
    Inspecting different tensors content in different points of the model code
    requires many executions and typically causes performance issues.
    Adding a step closure will ensure that it will be run after the barrier, when
    all the live tensors will be already materialized to device data.
    Live tensors which will include the ones captured by the closure arguments.
    So using `add_step_closure()` will ensure a single execution will be
    performed, even when multiple closures are queued, requiring multiple tensors
    to be inspected.
    Step closures will be run sequentially in the order they have been queued.
    Note that even though using this API the execution will be optimized, it is
    advised to throttle the printing/reporting events once every N steps.
    Args:
      closure (callable): The function to be called.
      args (tuple): The arguments to be passed to the closure.
      run_async: If True, run the closure asynchronously.
    """
    devctx = get_device_context()
    closures_type = 'async_step_closures' if run_async else 'step_closures'
    step_closures = getattr(devctx, closures_type, None)
    if step_closures is None:
        step_closures = []
        setattr(devctx, closures_type, step_closures)
    step_closures.append(lambda a=args: closure(*a))
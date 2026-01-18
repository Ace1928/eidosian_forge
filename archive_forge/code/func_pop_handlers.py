import inspect
from functools import partial
from weakref import WeakMethod
def pop_handlers(self):
    """Pop the top level of event handlers off the stack.
        """
    assert self._event_stack and 'No handlers pushed'
    del self._event_stack[0]
import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def getcontext():
    """Returns this thread's context.

    If this thread does not yet have a context, returns
    a new context and sets this thread's context.
    New contexts are copies of DefaultContext.
    """
    try:
        return _current_context_var.get()
    except LookupError:
        context = Context()
        _current_context_var.set(context)
        return context
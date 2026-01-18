import sys
import inspect
def getattr_(obj, name, default_thunk):
    """Similar to .setdefault in dictionaries."""
    try:
        return getattr(obj, name)
    except AttributeError:
        default = default_thunk()
        setattr(obj, name, default)
        return default
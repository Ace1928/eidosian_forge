from _pydevd_bundle.pydevd_comm import get_global_debugger
from _pydev_bundle import pydev_log
def replace_builtin_property(new_property=None):
    if new_property is None:
        new_property = DebugProperty
    original = property
    try:
        import builtins
        builtins.__dict__['property'] = new_property
    except:
        pydev_log.exception()
    return original
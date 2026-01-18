import sys
def extract_module_locals(depth=0):
    """Returns (module, locals) of the function `depth` frames away from the caller"""
    f = sys._getframe(depth + 1)
    global_ns = f.f_globals
    module = sys.modules[global_ns['__name__']]
    return (module, f.f_locals)
import sys
from _pydev_bundle import pydev_log
def is_interactive_backend(backend):
    """ Check if backend is interactive """
    matplotlib = sys.modules['matplotlib']
    from matplotlib.rcsetup import interactive_bk, non_interactive_bk
    if backend in interactive_bk:
        return True
    elif backend in non_interactive_bk:
        return False
    else:
        return matplotlib.is_interactive()
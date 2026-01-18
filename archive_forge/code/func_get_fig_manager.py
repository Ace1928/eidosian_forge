import atexit
from collections import OrderedDict
@classmethod
def get_fig_manager(cls, num):
    """
        If manager number *num* exists, make it the active one and return it;
        otherwise return *None*.
        """
    manager = cls.figs.get(num, None)
    if manager is not None:
        cls.set_active(manager)
    return manager
import atexit
from collections import OrderedDict
@classmethod
def get_all_fig_managers(cls):
    """Return a list of figure managers."""
    return list(cls.figs.values())
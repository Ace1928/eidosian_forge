import atexit
from collections import OrderedDict
@classmethod
def get_num_fig_managers(cls):
    """Return the number of figures being managed."""
    return len(cls.figs)
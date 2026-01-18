import atexit
from collections import OrderedDict
@classmethod
def has_fignum(cls, num):
    """Return whether figure number *num* exists."""
    return num in cls.figs
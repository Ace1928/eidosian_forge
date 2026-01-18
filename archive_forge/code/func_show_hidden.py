import re
import types
from IPython.utils.dir2 import dir2
def show_hidden(str, show_all=False):
    """Return true for strings starting with single _ if show_all is true."""
    return show_all or str.startswith('__') or (not str.startswith('_'))
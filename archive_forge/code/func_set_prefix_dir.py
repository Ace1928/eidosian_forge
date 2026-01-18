import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def set_prefix_dir(path):
    """Set the prefix used by the Enchant library to find its plugins

    Called automatically when the Python library is imported when
    required.
    """
    return _e.set_prefix_dir(path)
    set_prefix_dir._DOC_ERRORS = ['plugins']
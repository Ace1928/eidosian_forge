import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def get_loaded_plugin(name):
    """Retrieve an already loaded plugin.

    Returns None if there is no such plugin loaded
    """
    try:
        module = sys.modules[_MODULE_PREFIX + name]
    except KeyError:
        return None
    if module is None:
        return None
    return PlugIn(name, module)
import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def format_concise_plugin_list(state=None):
    """Return a string holding a concise list of plugins and their version.
    """
    if state is None:
        state = breezy.get_global_state()
    items = []
    for name, a_plugin in sorted(getattr(state, 'plugins', {}).items()):
        items.append('%s[%s]' % (name, a_plugin.__version__))
    return ', '.join(items)
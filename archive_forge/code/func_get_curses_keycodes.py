import re
import time
import platform
from collections import OrderedDict
import six
def get_curses_keycodes():
    """
    Return mapping of curses key-names paired by their keycode integer value.

    :rtype: dict
    :returns: Dictionary of (name, code) pairs for curses keyboard constant
        values and their mnemonic name. Such as code ``260``, with the value of
        its key-name identity, ``u'KEY_LEFT'``.
    """
    _keynames = [attr for attr in dir(curses) if attr.startswith('KEY_')]
    return {keyname: getattr(curses, keyname) for keyname in _keynames}
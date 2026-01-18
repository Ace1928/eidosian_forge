import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
def enable_vte():
    if _check_enabled('vte'):
        return
    gi.require_version('Vte', '0.0')
    from gi.repository import Vte
    _patch_module('vte', Vte)
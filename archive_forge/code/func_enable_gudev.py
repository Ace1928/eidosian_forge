import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
def enable_gudev():
    if _check_enabled('gudev'):
        return
    gi.require_version('GUdev', '1.0')
    from gi.repository import GUdev
    _patch_module('gudev', GUdev)
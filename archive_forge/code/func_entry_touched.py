from weakref import ref
from time import time
from kivy.core.text import DEFAULT_FONT
from kivy.compat import string_types
from kivy.factory import Factory
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.utils import platform as core_platform
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import (
import collections.abc
from os import listdir
from os.path import (
from fnmatch import fnmatch
def entry_touched(self, entry, touch):
    """(internal) This method must be called by the template when an entry
        is touched by the user.
        """
    if 'button' in touch.profile and touch.button in ('scrollup', 'scrolldown', 'scrollleft', 'scrollright'):
        return False
    _dir = self.file_system.is_dir(entry.path)
    dirselect = self.dirselect
    if _dir and dirselect and touch.is_double_tap:
        self.open_entry(entry)
        return
    if self.multiselect:
        if entry.path in self.selection:
            self.selection.remove(entry.path)
        else:
            if _dir and (not self.dirselect):
                self.open_entry(entry)
                return
            self.selection.append(entry.path)
    else:
        if _dir and (not self.dirselect):
            return
        self.selection = [abspath(join(self.path, entry.path))]
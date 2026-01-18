import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def inside_word(self, loc):
    if loc.inside_word():
        return True
    elif self.starts_word(loc):
        return True
    elif loc.ends_word() and (not self.ends_word(loc)):
        return True
    else:
        return False
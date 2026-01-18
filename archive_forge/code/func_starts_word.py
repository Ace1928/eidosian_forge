import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def starts_word(self, loc):
    if loc.starts_word():
        if loc.is_start():
            return True
        else:
            tmp = loc.copy()
            tmp.backward_char()
            return not self.is_extra_word_char(tmp)
    else:
        return False
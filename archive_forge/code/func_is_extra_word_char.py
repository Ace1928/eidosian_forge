import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def is_extra_word_char(self, loc):
    char = loc.get_char()
    return char != '' and char in self._extra_word_chars
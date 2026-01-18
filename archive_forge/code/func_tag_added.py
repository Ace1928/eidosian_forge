import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def tag_added(tag, *args):
    if hasattr(tag, 'spell_check') and (not tag.spell_check):
        self.ignored_tags.append(tag)
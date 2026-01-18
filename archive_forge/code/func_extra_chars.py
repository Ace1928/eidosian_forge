import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
@extra_chars.setter
def extra_chars(self, chars):
    """
        Set the list of extra characters beyond which words are extended.

        :param val: String containing list of characters
        """
    self._extra_chars = chars
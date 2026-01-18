import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def move_click_mark(self, iter):
    """
        Move the "click" mark, used to determine the word being checked.

        :param iter: TextIter for the new location
        """
    self._marks['click'].move(iter)
import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def forward_word_end(self, loc):

    def move_through_extra_chars():
        moved = False
        while self.is_extra_word_char(loc):
            if not loc.forward_char():
                break
            moved = True
        return moved
    tmp = loc.copy()
    tmp.backward_char()
    loc.forward_word_end()
    while move_through_extra_chars():
        if loc.is_end() or not loc.inside_word() or (not loc.forward_word_end()):
            break
import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def backward_word_start(self, loc):

    def move_through_extra_chars():
        tmp = loc.copy()
        tmp.backward_char()
        moved = False
        while self.is_extra_word_char(tmp):
            moved = True
            loc.assign(tmp)
            if not tmp.backward_char():
                break
        return moved
    loc.backward_word_start()
    while move_through_extra_chars():
        tmp = loc.copy()
        tmp.backward_char()
        if loc.is_start() or not tmp.inside_word() or (not loc.backward_word_start()):
            break
import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def check_range(self, start, end, force_all=False):
    """
        Checks a specified range between two GtkTextIters.

        :param start: Start iter - checking starts here.
        :param end: End iter - checking ends here.
        """
    logger.debug('Check range called with range %d:%d to %d:%d and force all set to %s.', start.get_line(), start.get_line_offset(), end.get_line(), end.get_line_offset(), force_all)
    if not self._enabled:
        return
    start = start.copy()
    end = end.copy()
    if self._iter_worker.inside_word(end):
        self._iter_worker.forward_word_end(end)
    if self._iter_worker.inside_word(start) or self._iter_worker.ends_word(start):
        self._iter_worker.backward_word_start(start)
    if not self._iter_worker.starts_word(start):
        self._iter_worker.forward_word_end(start)
        self._iter_worker.backward_word_start(start)
    self._buffer.remove_tag(self._misspelled, start, end)
    cursor = self._buffer.get_iter_at_mark(self._buffer.get_insert())
    precursor = cursor.copy()
    precursor.backward_char()
    highlight = cursor.has_tag(self._misspelled) or precursor.has_tag(self._misspelled)
    word_start = start.copy()
    while word_start.compare(end) < 0:
        word_end = word_start.copy()
        self._iter_worker.forward_word_end(word_end)
        in_word = word_start.compare(cursor) < 0 and cursor.compare(word_end) <= 0
        if in_word and (not force_all):
            if highlight:
                self._check_word(word_start, word_end)
            else:
                self._deferred_check = True
        else:
            self._check_word(word_start, word_end)
            self._deferred_check = False
        self._iter_worker.forward_word_end(word_end)
        self._iter_worker.backward_word_start(word_end)
        if word_start.equal(word_end):
            break
        word_start = word_end.copy()
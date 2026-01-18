import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def buffer_initialize(self):
    """
        Initialize the GtkTextBuffer associated with the GtkTextView. If you
        have associated a new GtkTextBuffer with the GtkTextView call this
        method.
        """
    self._misspelled = Gtk.TextTag.new('{}-misspelled'.format(self._prefix))
    self._misspelled.set_property('underline', 4)
    self._buffer = self._view.get_buffer()
    self._buffer.connect('insert-text', self._before_text_insert)
    self._buffer.connect_after('insert-text', self._after_text_insert)
    self._buffer.connect_after('delete-range', self._range_delete)
    self._buffer.connect_after('mark-set', self._mark_set)
    start = self._buffer.get_bounds()[0]
    self._marks = {'insert-start': SpellChecker._Mark(self._buffer, '{}-insert-start'.format(self._prefix), start, self._iter_worker), 'insert-end': SpellChecker._Mark(self._buffer, '{}-insert-end'.format(self._prefix), start, self._iter_worker), 'click': SpellChecker._Mark(self._buffer, '{}-click'.format(self._prefix), start, self._iter_worker)}
    self._table = self._buffer.get_tag_table()
    self._table.add(self._misspelled)
    self.ignored_tags = []

    def tag_added(tag, *args):
        if hasattr(tag, 'spell_check') and (not tag.spell_check):
            self.ignored_tags.append(tag)

    def tag_removed(tag, *args):
        if tag in self.ignored_tags:
            self.ignored_tags.remove(tag)
    self._table.connect('tag-added', tag_added)
    self._table.connect('tag-removed', tag_removed)
    self._table.foreach(tag_added, None)
    self.no_spell_check = self._table.lookup('no-spell-check')
    if not self.no_spell_check:
        self.no_spell_check = Gtk.TextTag.new('no-spell-check')
        self._table.add(self.no_spell_check)
    self.recheck()
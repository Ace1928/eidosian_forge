from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def selection_ranges(self):
    """
        Return a list of (from, to) tuples for the selection or none if nothing
        was selected.  start and end position are always included in the
        selection.

        This will yield several (from, to) tuples in case of a BLOCK selection.
        """
    if self.selection:
        from_, to = sorted([self.cursor_position, self.selection.original_cursor_position])
        if self.selection.type == SelectionType.BLOCK:
            from_line, from_column = self.translate_index_to_position(from_)
            to_line, to_column = self.translate_index_to_position(to)
            from_column, to_column = sorted([from_column, to_column])
            lines = self.lines
            for l in range(from_line, to_line + 1):
                line_length = len(lines[l])
                if from_column < line_length:
                    yield (self.translate_row_col_to_index(l, from_column), self.translate_row_col_to_index(l, min(line_length - 1, to_column)))
        else:
            if self.selection.type == SelectionType.LINES:
                from_ = max(0, self.text.rfind('\n', 0, from_) + 1)
                if self.text.find('\n', to) >= 0:
                    to = self.text.find('\n', to)
                else:
                    to = len(self.text) - 1
            yield (from_, to)
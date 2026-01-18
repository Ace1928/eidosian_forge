from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def selection_range_at_line(self, row):
    """
        If the selection spans a portion of the given line, return a (from, to) tuple.
        Otherwise, return None.
        """
    if self.selection:
        row_start = self.translate_row_col_to_index(row, 0)
        row_end = self.translate_row_col_to_index(row, max(0, len(self.lines[row]) - 1))
        from_, to = sorted([self.cursor_position, self.selection.original_cursor_position])
        intersection_start = max(row_start, from_)
        intersection_end = min(row_end, to)
        if intersection_start <= intersection_end:
            if self.selection.type == SelectionType.LINES:
                intersection_start = row_start
                intersection_end = row_end
            elif self.selection.type == SelectionType.BLOCK:
                _, col1 = self.translate_index_to_position(from_)
                _, col2 = self.translate_index_to_position(to)
                col1, col2 = sorted([col1, col2])
                intersection_start = self.translate_row_col_to_index(row, col1)
                intersection_end = self.translate_row_col_to_index(row, col2)
            _, from_column = self.translate_index_to_position(intersection_start)
            _, to_column = self.translate_index_to_position(intersection_end)
            return (from_column, to_column)
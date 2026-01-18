import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def column_index(self, i):
    """
        If ``i`` is a valid column index integer, then return it as is.
        Otherwise, check if ``i`` is used as the name for any column;
        if so, return that column's index.  Otherwise, raise a
        ``KeyError`` exception.
        """
    if isinstance(i, int) and 0 <= i < self._num_columns:
        return i
    else:
        return self._column_name_to_index[i]
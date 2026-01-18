import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def itemconfigure(self, row_index, col_index, cnf=None, **kw):
    """:see: ``MultiListbox.itemconfigure()``"""
    col_index = self.column_index(col_index)
    return self._mlb.itemconfigure(row_index, col_index, cnf, **kw)
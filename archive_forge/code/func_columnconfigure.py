import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def columnconfigure(self, col_index, cnf={}, **kw):
    """:see: ``MultiListbox.columnconfigure()``"""
    col_index = self.column_index(col_index)
    self._mlb.columnconfigure(col_index, cnf, **kw)
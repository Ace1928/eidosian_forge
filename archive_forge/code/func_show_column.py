import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def show_column(self, column_index):
    """:see: ``MultiListbox.show_column()``"""
    self._mlb.show_column(self.column_index(column_index))
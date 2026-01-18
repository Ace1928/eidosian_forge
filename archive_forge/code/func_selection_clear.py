import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def selection_clear(self, *args, **kwargs):
    for lb in self._listboxes:
        lb.selection_clear(*args, **kwargs)
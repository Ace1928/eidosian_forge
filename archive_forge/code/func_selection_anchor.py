import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def selection_anchor(self, *args, **kwargs):
    for lb in self._listboxes:
        lb.selection_anchor(*args, **kwargs)
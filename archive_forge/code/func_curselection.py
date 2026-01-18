import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def curselection(self, *args, **kwargs):
    return self._listboxes[0].curselection(*args, **kwargs)
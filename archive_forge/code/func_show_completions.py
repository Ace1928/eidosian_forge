import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def show_completions(self, comps):
    self.text.delete(self.tab_index, Tk_.END)
    width = self.text.winfo_width()
    font = Font(self.text, self.text.cget('font'))
    charwidth = width // self.char_size
    biggest = 2 + max([len(x) for x in comps])
    num_cols = charwidth // biggest
    num_rows = (len(comps) + num_cols - 1) // num_cols
    rows = []
    format = '%%-%ds' % biggest
    for n in range(num_rows):
        rows.append(''.join((format % x for x in comps[n:len(comps):num_rows])))
    view = '\n'.join(rows)
    self.text.insert(self.tab_index, '\n' + view)
    self.text.mark_set(Tk_.INSERT, self.tab_index)
    self.text.see(Tk_.END)
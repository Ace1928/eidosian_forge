import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def clear_completions(self):
    if self.tab_index:
        self.text.delete(self.tab_index, Tk_.END)
        self.tab_index = None
        self.tab_count = 0
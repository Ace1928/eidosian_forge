import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def middle_mouse_up(self, event):
    if self.nasty:
        start = self.text.search(self.nasty_text, index=self.nasty + '-2c')
        if start:
            self.text.delete(start, Tk_.INSERT)
        self.nasty = None
        self.nasty_text = None
    try:
        self.text.tag_remove(Tk_.SEL, Tk_.SEL_FIRST, Tk_.SEL_LAST)
    except Tk_.TclError:
        pass
    return 'break'
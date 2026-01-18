import string
import tkinter as Tkinter
import tkinter.font as tkFont
from . import ansi
def downPressed(self, event):
    self.callback('\x1bOB')
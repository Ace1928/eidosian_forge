import itertools
import re
from tkinter import SEL_FIRST, SEL_LAST, Frame, Label, PhotoImage, Scrollbar, Text, Tk
def initScrollText(self, frm, txt, contents):
    scl = Scrollbar(frm)
    scl.config(command=txt.yview)
    scl.pack(side='right', fill='y')
    txt.pack(side='left', expand=True, fill='x')
    txt.config(yscrollcommand=scl.set)
    txt.insert('1.0', contents)
    frm.pack(fill='x')
    Frame(height=2, bd=1, relief='ridge').pack(fill='x')
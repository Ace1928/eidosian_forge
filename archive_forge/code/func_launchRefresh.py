import itertools
import re
from tkinter import SEL_FIRST, SEL_LAST, Frame, Label, PhotoImage, Scrollbar, Text, Tk
def launchRefresh(_):
    sz.fld.after_idle(sz.refresh)
    rz.fld.after_idle(rz.refresh)
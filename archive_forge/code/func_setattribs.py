import tkinter
from tkinter.constants import *
def setattribs(element, **kwargs):
    for k, v in kwargs.items():
        element.setAttribute(k, str(v))
    return element
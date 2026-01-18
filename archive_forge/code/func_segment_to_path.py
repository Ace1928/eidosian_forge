import tkinter
from tkinter.constants import *
def segment_to_path(document, coords):
    """polyline with 2 vertices using <path> tag"""
    return setattribs(document.createElement('path'), d='M%s,%s %s,%s' % tuple(coords[:4]))
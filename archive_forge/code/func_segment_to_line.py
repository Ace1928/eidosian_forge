import tkinter
from tkinter.constants import *
def segment_to_line(document, coords):
    """polyline with 2 vertices using <line> tag"""
    return setattribs(document.createElement('line'), x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])
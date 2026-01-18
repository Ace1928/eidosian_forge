from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingRecursiveDescentParser
from nltk.tree import Tree
from nltk.util import in_idle
def autostep(self, *e):
    if self._animation_frames.get() == 0:
        self._animation_frames.set(2)
    if self._autostep:
        self._autostep = 0
    else:
        self._autostep = 1
        self._step()
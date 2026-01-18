from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def bind_click_leaves(self, callback, button=1):
    """
        Add a binding to all leaves.
        """
    for leaf in self._leaves:
        leaf.bind_click(callback, button)
    for leaf in self._leaves:
        leaf.bind_click(callback, button)
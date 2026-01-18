from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def bind_drag_nodes(self, callback, button=1):
    """
        Add a binding to all nodes.
        """
    for node in self._nodes:
        node.bind_drag(callback, button)
    for node in self._nodes:
        node.bind_drag(callback, button)
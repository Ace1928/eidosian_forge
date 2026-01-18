import os.path
import pickle
from tkinter import (
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.font import Font
from tkinter.messagebox import showerror, showinfo
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal
from nltk.parse.chart import (
from nltk.tree import Tree
from nltk.util import in_idle
def select_edge(self, edge):
    if edge in self._left_chart:
        self._left_matrix.markonly_edge(edge)
    else:
        self._left_matrix.unmark_edge()
    if edge in self._right_chart:
        self._right_matrix.markonly_edge(edge)
    else:
        self._right_matrix.unmark_edge()
    if edge in self._out_chart:
        self._out_matrix.markonly_edge(edge)
    else:
        self._out_matrix.unmark_edge()
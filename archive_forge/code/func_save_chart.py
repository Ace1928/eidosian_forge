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
def save_chart(self, *args):
    """Save a chart to a pickle file"""
    filename = asksaveasfilename(filetypes=self.CHART_FILE_TYPES, defaultextension='.pickle')
    if not filename:
        return
    try:
        with open(filename, 'wb') as outfile:
            pickle.dump(self._chart, outfile)
    except Exception as e:
        raise
        showerror('Error Saving Chart', 'Unable to open file: %r' % filename)
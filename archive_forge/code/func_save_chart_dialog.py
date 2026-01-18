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
def save_chart_dialog(self, *args):
    filename = asksaveasfilename(filetypes=self.CHART_FILE_TYPES, defaultextension='.pickle')
    if not filename:
        return
    try:
        with open(filename, 'wb') as outfile:
            pickle.dump(self._out_chart, outfile)
    except Exception as e:
        showerror('Error Saving Chart', f'Unable to open file: {filename!r}\n{e}')
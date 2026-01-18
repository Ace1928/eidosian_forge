from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingShiftReduceParser
from nltk.tree import Tree
from nltk.util import in_idle
def edit_sentence(self, *e):
    sentence = ' '.join(self._sent)
    title = 'Edit Text'
    instr = 'Enter a new sentence to parse.'
    EntryDialog(self._top, sentence, instr, self.set_sentence, title)
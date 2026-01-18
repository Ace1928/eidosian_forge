import random
import re
import textwrap
import time
from tkinter import (
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.font import Font
from nltk.chunk import ChunkScore, RegexpChunkParser
from nltk.chunk.regexp import RegexpChunkRule
from nltk.corpus import conll2000, treebank_chunk
from nltk.draw.util import ShowText
from nltk.tree import Tree
from nltk.util import in_idle
def show_devset(self, index=None):
    if index is None:
        index = self.devset_index
    index = min(max(0, index), self._devset_size.get() - 1)
    if index == self.devset_index and (not self._showing_trace):
        return
    self.devset_index = index
    self._showing_trace = False
    self.trace_button['state'] = 'normal'
    self.devset_button['state'] = 'disabled'
    self.devsetbox['state'] = 'normal'
    self.devsetbox['wrap'] = 'word'
    self.devsetbox.delete('1.0', 'end')
    self.devsetlabel['text'] = 'Development Set (%d/%d)' % (self.devset_index + 1, self._devset_size.get())
    sample = self.devset[self.devset_index:self.devset_index + 1]
    self.charnum = {}
    self.linenum = {0: 1}
    for sentnum, sent in enumerate(sample):
        linestr = ''
        for wordnum, (word, pos) in enumerate(sent.leaves()):
            self.charnum[sentnum, wordnum] = len(linestr)
            linestr += f'{word}/{pos} '
            self.charnum[sentnum, wordnum + 1] = len(linestr)
        self.devsetbox.insert('end', linestr[:-1] + '\n\n')
    if self.chunker is not None:
        self._highlight_devset()
    self.devsetbox['state'] = 'disabled'
    first = self.devset_index / self._devset_size.get()
    last = (self.devset_index + 2) / self._devset_size.get()
    self.devset_scroll.set(first, last)
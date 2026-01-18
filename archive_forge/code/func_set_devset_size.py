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
def set_devset_size(self, size=None):
    if size is not None:
        self._devset_size.set(size)
    self._devset_size.set(min(len(self.devset), self._devset_size.get()))
    self.show_devset(1)
    self.show_devset(0)
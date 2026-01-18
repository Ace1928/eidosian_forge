import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def corpus_selected(self, *args):
    new_selection = self.var.get()
    self.load_corpus(new_selection)
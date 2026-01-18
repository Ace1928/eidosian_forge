import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def non_default_corpora(self):
    copy = []
    copy.extend(list(self.CORPORA.keys()))
    copy.remove(self.DEFAULT_CORPUS)
    copy.sort()
    return copy
import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def freeze_editable(self):
    self.query_box['state'] = 'disabled'
    self.search_button['state'] = 'disabled'
    self.prev['state'] = 'disabled'
    self.next['state'] = 'disabled'
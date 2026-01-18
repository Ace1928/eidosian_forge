import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def set_paging_button_states(self):
    if self.current_page == 0 or self.current_page == 1:
        self.prev['state'] = 'disabled'
    else:
        self.prev['state'] = 'normal'
    if self.model.has_more_pages(self.current_page):
        self.next['state'] = 'normal'
    else:
        self.next['state'] = 'disabled'
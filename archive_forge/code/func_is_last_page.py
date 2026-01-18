import queue as q
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.probability import FreqDist
from nltk.util import in_idle
def is_last_page(self, number):
    if number < len(self.result_pages):
        return False
    return self.results_returned + (number - len(self.result_pages)) * self.result_count >= len(self.collocations)
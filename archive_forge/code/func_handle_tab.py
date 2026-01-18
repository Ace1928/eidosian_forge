import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def handle_tab(self, event):
    if self.text.compare(Tk_.INSERT, '<', 'output_end'):
        return 'break'
    self.tab_index = self.text.index(Tk_.INSERT)
    self.tab_count += 1
    if self.tab_count > 2:
        self.clear_completions()
        return 'break'
    line = self.text.get('output_end', self.tab_index).strip('\n')
    word = delims.split(line)[-1]
    try:
        completions = self.IP.complete(word)[1]
    except TypeError:
        completions = []
    if word.find('_') == -1:
        completions = [x for x in completions if x.find('__') == -1 and x.find('._') == -1]
    if len(completions) == 0:
        self.window.bell()
        self.tab_count = 0
        return 'break'
    stem = self.stem(completions)
    if len(stem) > len(word):
        self.do_completion(word, stem)
    elif len(completions) > 60 and self.tab_count == 1:
        self.show_completions(['%s possibilities -- hit tab again to view them all' % len(completions)])
    else:
        self.show_completions(completions)
        if len(completions) <= 60:
            self.tab_count += 1
    return 'break'
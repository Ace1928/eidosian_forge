import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def handle_up(self, event, jump=False):
    if self.text.compare(Tk_.INSERT, '<', 'output_end'):
        return
    insert_line_number = str(self.text.index(Tk_.INSERT)).split('.')[0]
    prompt_line_number = str(self.text.index('output_end')).split('.')[0]
    if insert_line_number != prompt_line_number:
        return
    if self.hist_pointer == 0:
        input_history = self.IP.history_manager.input_hist_raw
        self.hist_stem = self.text.get('output_end', Tk_.END).strip()
        self.filtered_hist = [x for x in input_history if x and x.startswith(self.hist_stem)]
    if self.hist_pointer >= len(self.filtered_hist):
        self.window.bell()
        return 'break'
    self.text.delete('output_end', Tk_.END)
    self.hist_pointer += 1
    self.write_history()
    if jump:
        self.text.mark_set(Tk_.INSERT, 'output_end')
    return 'break'
import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def interact_handle_input(self, cell, script=False):
    """
        Validate the code in the cell.  If the code is valid but
        incomplete, set the indent that should follow a continuation
        prompt and set the 'more' flag.  If the code is valid and
        complete then run the code.
        """
    transformer = self.IP.input_transformer_manager
    assert cell.endswith('\n')
    if not cell.strip():
        self._current_indent = 0
        return
    if script:
        self._input_buffer += cell
    else:
        self._input_buffer = self.clean_code(cell)
    status, indent = transformer.check_complete(self._input_buffer)
    self._current_indent = indent or 0
    if status == 'incomplete':
        self.IP.more = True
        return
    if status == 'invalid':
        self.text.mark_set(Tk_.INSERT, self.text.index('output_end-1line'))
        self.IP.run_cell(self._input_buffer, store_history=True)
        self.text.insert('output_end-1line', '\n')
        self.reset()
        self.text.delete('output_end', Tk_.END)
        return
    insert_line = int(self.text.index(Tk_.INSERT).split('.')[0])
    prompt_line = int(self.text.index('output_end').split('.')[0])
    tail = self.text.get('%d.%d' % (insert_line, self._prompt_size), Tk_.END)
    if not tail.strip():
        self.text.tag_delete('history')
        self._input_buffer = self._input_buffer.rstrip() + '\n'
        self.text.delete(Tk_.INSERT, Tk_.END)
        self.text.insert(Tk_.INSERT, '\n')
        self.text.mark_set('output_end', Tk_.INSERT)
        self.multiline = False
        self.running_code = True
        self.editing_hist = False
        last_line = insert_line - 1
        if last_line > prompt_line:
            self.text.delete('%d.0' % last_line, '%d.0 lineend' % last_line)
        self.IP.run_cell(self._input_buffer, store_history=True)
        self.write('\n')
        self.text.tag_add('output', 'output_end', Tk_.END)
        self.reset()
    else:
        self.IP.more = True
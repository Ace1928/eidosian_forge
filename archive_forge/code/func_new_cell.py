from __future__ import annotations
import re
from .nbbase import new_code_cell, new_notebook, new_text_cell, new_worksheet
from .rwbase import NotebookReader, NotebookWriter
def new_cell(self, state, lines):
    """Create a new cell."""
    if state == 'codecell':
        input_ = '\n'.join(lines)
        input_ = input_.strip('\n')
        if input_:
            return new_code_cell(input=input_)
    elif state == 'htmlcell':
        text = self._remove_comments(lines)
        if text:
            return new_text_cell('html', source=text)
    elif state == 'markdowncell':
        text = self._remove_comments(lines)
        if text:
            return new_text_cell('markdown', source=text)
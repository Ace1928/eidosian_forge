from __future__ import annotations
import re
from .nbbase import new_code_cell, new_notebook, new_text_cell, new_worksheet
from .rwbase import NotebookReader, NotebookWriter
def split_lines_into_blocks(self, lines):
    """Split lines into code blocks."""
    if len(lines) == 1:
        yield lines[0]
        raise StopIteration()
    import ast
    source = '\n'.join(lines)
    code = ast.parse(source)
    starts = [x.lineno - 1 for x in code.body]
    for i in range(len(starts) - 1):
        yield '\n'.join(lines[starts[i]:starts[i + 1]]).strip('\n')
    yield '\n'.join(lines[starts[-1]:]).strip('\n')
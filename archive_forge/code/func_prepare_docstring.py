import re
import sys
import warnings
from typing import Dict, List, Tuple
from docutils.parsers.rst.states import Body
from sphinx.deprecation import RemovedInSphinx60Warning
def prepare_docstring(s: str, tabsize: int=8) -> List[str]:
    """Convert a docstring into lines of parseable reST.  Remove common leading
    indentation, where the indentation of the first line is ignored.

    Return the docstring as a list of lines usable for inserting into a docutils
    ViewList (used as argument of nested_parse().)  An empty line is added to
    act as a separator between this docstring and following content.
    """
    lines = s.expandtabs(tabsize).splitlines()
    margin = sys.maxsize
    for line in lines[1:]:
        content = len(line.lstrip())
        if content:
            indent = len(line) - content
            margin = min(margin, indent)
    if len(lines):
        lines[0] = lines[0].lstrip()
    if margin < sys.maxsize:
        for i in range(1, len(lines)):
            lines[i] = lines[i][margin:]
    while lines and (not lines[0]):
        lines.pop(0)
    if lines and lines[-1]:
        lines.append('')
    return lines
import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def get_2D_block(self, top, left, bottom, right, strip_indent=True):
    block = self[top:bottom]
    indent = right
    for i in range(len(block.data)):
        ci = utils.column_indices(block.data[i])
        try:
            left = ci[left]
        except IndexError:
            left += len(block.data[i]) - len(ci)
        try:
            right = ci[right]
        except IndexError:
            right += len(block.data[i]) - len(ci)
        block.data[i] = line = block.data[i][left:right].rstrip()
        if line:
            indent = min(indent, len(line) - len(line.lstrip()))
    if strip_indent and 0 < indent < right:
        block.data = [line[indent:] for line in block.data]
    return block
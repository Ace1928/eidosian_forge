import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def get_indented(self, start=0, until_blank=False, strip_indent=True, block_indent=None, first_indent=None):
    """
        Extract and return a StringList of indented lines of text.

        Collect all lines with indentation, determine the minimum indentation,
        remove the minimum indentation from all indented lines (unless
        `strip_indent` is false), and return them. All lines up to but not
        including the first unindented line will be returned.

        :Parameters:
          - `start`: The index of the first line to examine.
          - `until_blank`: Stop collecting at the first blank line if true.
          - `strip_indent`: Strip common leading indent if true (default).
          - `block_indent`: The indent of the entire block, if known.
          - `first_indent`: The indent of the first line, if known.

        :Return:
          - a StringList of indented lines with mininum indent removed;
          - the amount of the indent;
          - a boolean: did the indented block finish with a blank line or EOF?
        """
    indent = block_indent
    end = start
    if block_indent is not None and first_indent is None:
        first_indent = block_indent
    if first_indent is not None:
        end += 1
    last = len(self.data)
    while end < last:
        line = self.data[end]
        if line and (line[0] != ' ' or (block_indent is not None and line[:block_indent].strip())):
            blank_finish = end > start and (not self.data[end - 1].strip())
            break
        stripped = line.lstrip()
        if not stripped:
            if until_blank:
                blank_finish = 1
                break
        elif block_indent is None:
            line_indent = len(line) - len(stripped)
            if indent is None:
                indent = line_indent
            else:
                indent = min(indent, line_indent)
        end += 1
    else:
        blank_finish = 1
    block = self[start:end]
    if first_indent is not None and block:
        block.data[0] = block.data[0][first_indent:]
    if indent and strip_indent:
        block.trim_left(indent, start=first_indent is not None)
    return (block, indent or 0, blank_finish)
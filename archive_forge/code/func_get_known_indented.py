import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def get_known_indented(self, indent, until_blank=False, strip_indent=True):
    """
        Return an indented block and info.

        Extract an indented block where the indent is known for all lines.
        Starting with the current line, extract the entire text block with at
        least `indent` indentation (which must be whitespace, except for the
        first line).

        :Parameters:
            - `indent`: The number of indent columns/characters.
            - `until_blank`: Stop collecting at the first blank line if true.
            - `strip_indent`: Strip `indent` characters of indentation if true
              (default).

        :Return:
            - the indented block,
            - its first line offset from BOF, and
            - whether or not it finished with a blank line.
        """
    offset = self.abs_line_offset()
    indented, indent, blank_finish = self.input_lines.get_indented(self.line_offset, until_blank, strip_indent, block_indent=indent)
    self.next_line(len(indented) - 1)
    while indented and (not indented[0].strip()):
        indented.trim_start()
        offset += 1
    return (indented, offset, blank_finish)
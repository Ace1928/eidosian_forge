import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def is_next_line_blank(self):
    """Return 1 if the next line is blank or non-existant."""
    try:
        return not self.input_lines[self.line_offset + 1].strip()
    except IndexError:
        return 1
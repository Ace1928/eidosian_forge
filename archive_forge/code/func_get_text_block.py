import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def get_text_block(self, start, flush_left=False):
    """
        Return a contiguous block of text.

        If `flush_left` is true, raise `UnexpectedIndentationError` if an
        indented line is encountered before the text block ends (with a blank
        line).
        """
    end = start
    last = len(self.data)
    while end < last:
        line = self.data[end]
        if not line.strip():
            break
        if flush_left and line[0] == ' ':
            source, offset = self.info(end)
            raise UnexpectedIndentationError(self[start:end], source, offset + 1)
        end += 1
    return self[start:end]
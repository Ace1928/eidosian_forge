import numbers
import random
import re
import string
import textwrap
def compact_empty_lines(text):
    """
    Replace repeating empty lines with a single empty line (similar to ``cat -s``).

    :param text: The text in which to compact empty lines (a string).
    :returns: The text with empty lines compacted (a string).
    """
    i = 0
    lines = text.splitlines(True)
    while i < len(lines):
        if i > 0 and is_empty_line(lines[i - 1]) and is_empty_line(lines[i]):
            lines.pop(i)
        else:
            i += 1
    return ''.join(lines)
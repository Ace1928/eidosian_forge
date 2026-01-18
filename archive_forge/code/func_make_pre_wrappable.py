import re
import html
from paste.util import PySourceColor
def make_pre_wrappable(html, wrap_limit=60, split_on=';?&@!$#-/\\"\''):
    """
    Like ``make_wrappable()`` but intended for text that will
    go in a ``<pre>`` block, so wrap on a line-by-line basis.
    """
    lines = html.splitlines()
    new_lines = []
    for line in lines:
        if len(line) > wrap_limit:
            for char in split_on:
                if char in line:
                    parts = line.split(char)
                    line = '<wbr>'.join(parts)
                    break
        new_lines.append(line)
    return '\n'.join(lines)
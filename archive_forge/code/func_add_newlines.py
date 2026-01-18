import re
import lxml
import lxml.etree
from lxml.html.clean import Cleaner
def add_newlines(tag, context):
    if not guess_layout:
        return
    prev = context.prev
    if prev is _DOUBLE_NEWLINE:
        return
    if tag in double_newline_tags:
        context.prev = _DOUBLE_NEWLINE
        chunks.append('\n' if prev is _NEWLINE else '\n\n')
    elif tag in newline_tags:
        context.prev = _NEWLINE
        if prev is not _NEWLINE:
            chunks.append('\n')
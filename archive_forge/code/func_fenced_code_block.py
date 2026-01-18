from __future__ import absolute_import, unicode_literals
import re
from commonmark import common
from commonmark.common import unescape_string
from commonmark.inlines import InlineParser
from commonmark.node import Node
@staticmethod
def fenced_code_block(parser, container=None):
    if not parser.indented:
        m = re.search(reCodeFence, parser.current_line[parser.next_nonspace:])
        if m:
            fence_length = len(m.group())
            parser.close_unmatched_blocks()
            container = parser.add_child('code_block', parser.next_nonspace)
            container.is_fenced = True
            container.fence_length = fence_length
            container.fence_char = m.group()[0]
            container.fence_offset = parser.indent
            parser.advance_next_nonspace()
            parser.advance_offset(fence_length, False)
            return 2
    return 0
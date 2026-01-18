from __future__ import absolute_import, unicode_literals
import re
from commonmark import common
from commonmark.common import unescape_string
from commonmark.inlines import InlineParser
from commonmark.node import Node
@staticmethod
def atx_heading(parser, container=None):
    if not parser.indented:
        m = re.search(reATXHeadingMarker, parser.current_line[parser.next_nonspace:])
        if m:
            parser.advance_next_nonspace()
            parser.advance_offset(len(m.group()), False)
            parser.close_unmatched_blocks()
            container = parser.add_child('heading', parser.next_nonspace)
            container.level = len(m.group().strip())
            container.string_content = re.sub('[ \\t]+#+[ \\t]*$', '', re.sub('^[ \\t]*#+[ \\t]*$', '', parser.current_line[parser.offset:]))
            parser.advance_offset(len(parser.current_line) - parser.offset, False)
            return 2
    return 0
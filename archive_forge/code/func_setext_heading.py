from __future__ import absolute_import, unicode_literals
import re
from commonmark import common
from commonmark.common import unescape_string
from commonmark.inlines import InlineParser
from commonmark.node import Node
@staticmethod
def setext_heading(parser, container=None):
    if not parser.indented and container.t == 'paragraph':
        m = re.search(reSetextHeadingLine, parser.current_line[parser.next_nonspace:])
        if m:
            parser.close_unmatched_blocks()
            while peek(container.string_content, 0) == '[':
                pos = parser.inline_parser.parseReference(container.string_content, parser.refmap)
                if not pos:
                    break
                container.string_content = container.string_content[pos:]
            if container.string_content:
                heading = Node('heading', container.sourcepos)
                heading.level = 1 if m.group()[0] == '=' else 2
                heading.string_content = container.string_content
                container.insert_after(heading)
                container.unlink()
                parser.tip = heading
                parser.advance_offset(len(parser.current_line) - parser.offset, False)
                return 2
            else:
                return 0
    return 0
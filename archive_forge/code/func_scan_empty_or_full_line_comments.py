from __future__ import annotations
from ruamel.yaml.error import MarkedYAMLError, CommentMark  # NOQA
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.docinfo import Version, Tag  # NOQA
from ruamel.yaml.compat import check_anchorname_char, _debug, nprint, nprintf  # NOQA
def scan_empty_or_full_line_comments(self) -> None:
    blmark = self.reader.get_mark()
    assert blmark.column == 0
    blanks = ''
    comment = None
    mark = None
    ch = self.reader.peek()
    while True:
        if ch in '\r\n\x85\u2028\u2029':
            if self.reader.prefix(2) == '\r\n':
                self.reader.forward(2)
            else:
                self.reader.forward()
            if comment is not None:
                comment += '\n'
                self.comments.add_full_line_comment(comment, mark.column, mark.line)
                comment = None
            else:
                blanks += '\n'
                self.comments.add_blank_line(blanks, blmark.column, blmark.line)
            blanks = ''
            blmark = self.reader.get_mark()
            ch = self.reader.peek()
            continue
        if comment is None:
            if ch in ' \t':
                blanks += ch
            elif ch == '#':
                mark = self.reader.get_mark()
                comment = '#'
            else:
                break
        else:
            comment += ch
        self.reader.forward()
        ch = self.reader.peek()
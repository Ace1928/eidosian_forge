import re
from pygments.lexer import RegexLexer, include
from pygments.lexers import get_lexer_for_mimetype
from pygments.token import Text, Name, String, Operator, Comment, Other
from pygments.util import get_int_opt, ClassNotFound
def get_body_tokens(self, match):
    pos_body_start = match.start()
    entire_body = match.group()
    if entire_body[0] == '\n':
        yield (pos_body_start, Text.Whitespace, '\n')
        pos_body_start = pos_body_start + 1
        entire_body = entire_body[1:]
    if not self.content_type.startswith('multipart') or not self.boundary:
        for i, t, v in self.get_bodypart_tokens(entire_body):
            yield (pos_body_start + i, t, v)
        return
    bdry_pattern = '^--%s(--)?\\n' % re.escape(self.boundary)
    bdry_matcher = re.compile(bdry_pattern, re.MULTILINE)
    m = bdry_matcher.search(entire_body)
    if m:
        pos_part_start = pos_body_start + m.end()
        pos_iter_start = lpos_end = m.end()
        yield (pos_body_start, Text, entire_body[:m.start()])
        yield (pos_body_start + lpos_end, String.Delimiter, m.group())
    else:
        pos_part_start = pos_body_start
        pos_iter_start = 0
    for m in bdry_matcher.finditer(entire_body, pos_iter_start):
        lpos_start = pos_part_start - pos_body_start
        lpos_end = m.start()
        part = entire_body[lpos_start:lpos_end]
        for i, t, v in self.get_bodypart_tokens(part):
            yield (pos_part_start + i, t, v)
        yield (pos_body_start + lpos_end, String.Delimiter, m.group())
        pos_part_start = pos_body_start + m.end()
    lpos_start = pos_part_start - pos_body_start
    if lpos_start != len(entire_body):
        yield (pos_part_start, Text, entire_body[lpos_start:])
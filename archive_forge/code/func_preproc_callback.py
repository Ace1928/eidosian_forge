import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def preproc_callback(self, match, ctx):
    proc = match.group(2)
    if proc == 'if':
        self.preproc_stack.append(ctx.stack[:])
    elif proc in ['else', 'elseif']:
        if self.preproc_stack:
            ctx.stack = self.preproc_stack[-1][:]
    elif proc == 'end':
        if self.preproc_stack:
            self.preproc_stack.pop()
    if proc in ['if', 'elseif']:
        ctx.stack.append('preproc-expr')
    if proc in ['error']:
        ctx.stack.append('preproc-error')
    yield (match.start(), Comment.Preproc, '#' + proc)
    ctx.pos = match.end()
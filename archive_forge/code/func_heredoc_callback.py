import re
from pygments.lexer import Lexer, RegexLexer, ExtendedRegexLexer, include, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
def heredoc_callback(self, match, ctx):
    start = match.start(1)
    yield (start, Operator, match.group(1))
    yield (match.start(2), String.Heredoc, match.group(2))
    yield (match.start(3), String.Delimiter, match.group(3))
    yield (match.start(4), String.Heredoc, match.group(4))
    heredocstack = ctx.__dict__.setdefault('heredocstack', [])
    outermost = not bool(heredocstack)
    heredocstack.append((match.group(1) == '<<-', match.group(3)))
    ctx.pos = match.start(5)
    ctx.end = match.end(5)
    for i, t, v in self.get_tokens_unprocessed(context=ctx):
        yield (i, t, v)
    ctx.pos = match.end()
    if outermost:
        for tolerant, hdname in heredocstack:
            lines = []
            for match in line_re.finditer(ctx.text, ctx.pos):
                if tolerant:
                    check = match.group().strip()
                else:
                    check = match.group().rstrip()
                if check == hdname:
                    for amatch in lines:
                        yield (amatch.start(), String.Heredoc, amatch.group())
                    yield (match.start(), String.Delimiter, match.group())
                    ctx.pos = match.end()
                    break
                else:
                    lines.append(match)
            else:
                for amatch in lines:
                    yield (amatch.start(), Error, amatch.group())
        ctx.end = len(ctx.text)
        del heredocstack[:]
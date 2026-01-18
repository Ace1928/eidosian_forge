import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
def opening_brace_callback(lexer, match, context):
    stack = context.stack
    yield (match.start(), Text, context.text[match.start():match.end()])
    context.pos = match.end()
    if len(stack) > 2 and stack[-2] == 'token':
        context.perl6_token_nesting_level += 1
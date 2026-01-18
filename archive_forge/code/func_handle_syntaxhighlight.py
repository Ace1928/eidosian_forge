import re
from pygments.lexers.html import XmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.css import CssLexer
from pygments.lexers.lilypond import LilyPondLexer
from pygments.lexers.data import JsonLexer
from pygments.lexer import RegexLexer, DelegatingLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, ClassNotFound
def handle_syntaxhighlight(self, match, ctx):
    from pygments.lexers import get_lexer_by_name
    attr_content = match.group()
    start = 0
    index = 0
    while True:
        index = attr_content.find('>', start)
        if attr_content[index - 2:index] != '--':
            break
        start = index + 1
    if index == -1:
        yield from self.get_tokens_unprocessed(attr_content, stack=['root', 'attr'])
        return
    attr = attr_content[:index]
    yield from self.get_tokens_unprocessed(attr, stack=['root', 'attr'])
    yield (match.start(3) + index, Punctuation, '>')
    lexer = None
    content = attr_content[index + 1:]
    lang_match = re.findall('\\blang=("|\\\'|)(\\w+)(\\1)', attr)
    if len(lang_match) >= 1:
        lang = lang_match[-1][1]
        try:
            lexer = get_lexer_by_name(lang)
        except ClassNotFound:
            pass
    if lexer is None:
        yield (match.start() + index + 1, Text, content)
    else:
        yield from lexer.get_tokens_unprocessed(content)
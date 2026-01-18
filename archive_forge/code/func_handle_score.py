import re
from pygments.lexers.html import XmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.css import CssLexer
from pygments.lexers.lilypond import LilyPondLexer
from pygments.lexers.data import JsonLexer
from pygments.lexer import RegexLexer, DelegatingLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, ClassNotFound
def handle_score(self, match, ctx):
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
    content = attr_content[index + 1:]
    yield from self.get_tokens_unprocessed(attr, stack=['root', 'attr'])
    yield (match.start(3) + index, Punctuation, '>')
    lang_match = re.findall('\\blang=("|\\\'|)(\\w+)(\\1)', attr)
    lang = lang_match[-1][1] if len(lang_match) >= 1 else 'lilypond'
    if lang == 'lilypond':
        yield from LilyPondLexer().get_tokens_unprocessed(content)
    else:
        yield (match.start() + index + 1, Text, content)
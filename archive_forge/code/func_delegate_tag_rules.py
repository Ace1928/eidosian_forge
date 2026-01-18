import re
from pygments.lexers.html import XmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.css import CssLexer
from pygments.lexers.lilypond import LilyPondLexer
from pygments.lexers.data import JsonLexer
from pygments.lexer import RegexLexer, DelegatingLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, ClassNotFound
def delegate_tag_rules(tag_name, lexer):
    return [('(?i)(</)({})(\\s*)(>)'.format(tag_name), bygroups(Punctuation, Name.Tag, Whitespace, Punctuation), '#pop'), ('(?si).+?(?=</{}\\s*>)'.format(tag_name), using(lexer))]
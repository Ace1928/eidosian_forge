from pygments.lexer import RegexLexer, DelegatingLexer, bygroups
from pygments.lexers.mime import MIMELexer
from pygments.token import Text, Keyword, Name, String, Number, Comment
from pygments.util import get_bool_opt
def get_x_header_tokens(self, match):
    if self.highlight_x:
        yield (match.start(1), Name.Tag, match.group(1))
        default_actions = self.get_tokens_unprocessed(match.group(2), stack=('root', 'header'))
        yield from default_actions
    else:
        yield (match.start(1), Comment.Special, match.group(1))
        yield (match.start(2), Comment.Multiline, match.group(2))
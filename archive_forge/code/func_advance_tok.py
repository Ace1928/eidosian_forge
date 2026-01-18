from __future__ import absolute_import, division, print_function
from . import lexer, error
from . import coretypes
def advance_tok(self):
    """Advances self.pos by one, if it is not already at the end."""
    if self.pos != self.end_pos:
        self.pos = self.pos + 1
        try:
            if self.pos >= len(self.tokens):
                self.tokens.append(next(self.lex))
        except StopIteration:
            if len(self.tokens) > 0:
                span = (self.tokens[self.pos - 1].span[1],) * 2
            else:
                span = (0, 0)
            self.tokens.append(lexer.Token(None, None, span, None))
            self.end_pos = self.pos
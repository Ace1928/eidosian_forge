import re
from .ply import lex
from .ply.lex import TOKEN
def find_tok_column(self, token):
    """ Find the column of the token in its line.
        """
    last_cr = self.lexer.lexdata.rfind('\n', 0, token.lexpos)
    return token.lexpos - last_cr
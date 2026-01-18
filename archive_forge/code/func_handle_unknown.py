import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def handle_unknown(self, current_token, preserve_statement_flags):
    self.print_token(current_token)
    if current_token.text[-1] == '\n':
        self.print_newline(preserve_statement_flags=preserve_statement_flags)
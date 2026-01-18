import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def handle_equals(self, current_token):
    if self.start_of_statement(current_token):
        pass
    else:
        self.handle_whitespace_and_comments(current_token)
    if self._flags.declaration_statement:
        self._flags.declaration_assignment = True
    self._output.space_before_token = True
    self.print_token(current_token)
    self._output.space_before_token = True
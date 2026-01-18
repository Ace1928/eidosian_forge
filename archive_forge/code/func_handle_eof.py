import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def handle_eof(self, current_token):
    while self._flags.mode == MODE.Statement:
        self.restore_mode()
    self.handle_whitespace_and_comments(current_token)
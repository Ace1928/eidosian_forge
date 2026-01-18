import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def print_newline(self, force_newline=False, preserve_statement_flags=False):
    if not preserve_statement_flags:
        if self._flags.last_token.text != ';' and self._flags.last_token.text != ',' and (self._flags.last_token.text != '=') and (self._flags.last_token.type != TOKEN.OPERATOR or self._flags.last_token.text == '--' or self._flags.last_token.text == '++'):
            next_token = self._tokens.peek()
            while self._flags.mode == MODE.Statement and (not (self._flags.if_block and reserved_word(next_token, 'else'))) and (not self._flags.do_block):
                self.restore_mode()
    if self._output.add_new_line(force_newline):
        self._flags.multiline_frame = True
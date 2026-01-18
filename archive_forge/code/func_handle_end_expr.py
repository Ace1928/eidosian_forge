import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def handle_end_expr(self, current_token):
    while self._flags.mode == MODE.Statement:
        self.restore_mode()
    self.handle_whitespace_and_comments(current_token)
    if self._flags.multiline_frame:
        self.allow_wrap_or_preserved_newline(current_token, current_token.text == ']' and self.is_array(self._flags.mode) and (not self._options.keep_array_indentation))
    if self._options.space_in_paren:
        if self._flags.last_token.type == TOKEN.START_EXPR and (not self._options.space_in_empty_paren):
            self._output.space_before_token = False
            self._output.trim()
        else:
            self._output.space_before_token = True
    self.deindent()
    self.print_token(current_token)
    self.restore_mode()
    remove_redundant_indentation(self._output, self._previous_flags)
    if self._flags.do_while and self._previous_flags.mode == MODE.Conditional:
        self._previous_flags.mode = MODE.Expression
        self._flags.do_block = False
        self._flags.do_while = False
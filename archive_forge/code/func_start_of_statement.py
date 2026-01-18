import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def start_of_statement(self, current_token):
    start = False
    start = start or (reserved_array(self._flags.last_token, ['var', 'let', 'const']) and current_token.type == TOKEN.WORD)
    start = start or reserved_word(self._flags.last_token, 'do')
    start = start or (not (self._flags.parent.mode == MODE.ObjectLiteral and self._flags.mode == MODE.Statement) and reserved_array(self._flags.last_token, self._newline_restricted_tokens) and (not current_token.newlines))
    start = start or (reserved_word(self._flags.last_token, 'else') and (not (reserved_word(current_token, 'if') and current_token.comments_before is None)))
    start = start or (self._flags.last_token.type == TOKEN.END_EXPR and (self._previous_flags.mode == MODE.ForInitializer or self._previous_flags.mode == MODE.Conditional))
    start = start or (self._flags.last_token.type == TOKEN.WORD and self._flags.mode == MODE.BlockStatement and (not self._flags.in_case) and (not (current_token.text == '--' or current_token.text == '++')) and (self._last_last_text != 'function') and (current_token.type != TOKEN.WORD) and (current_token.type != TOKEN.RESERVED))
    start = start or (self._flags.mode == MODE.ObjectLiteral and (self._flags.last_token.text == ':' and self._flags.ternary_depth == 0 or reserved_array(self._flags.last_token, ['get', 'set'])))
    if start:
        self.set_mode(MODE.Statement)
        self.indent()
        self.handle_whitespace_and_comments(current_token, True)
        if not self.start_of_object_property():
            self.allow_wrap_or_preserved_newline(current_token, reserved_array(current_token, ['do', 'for', 'if', 'while']))
        return True
    else:
        return False
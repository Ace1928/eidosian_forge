import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def handle_start_block(self, current_token):
    self.handle_whitespace_and_comments(current_token)
    next_token = self._tokens.peek()
    second_token = self._tokens.peek(1)
    if self._flags.last_word == 'switch' and self._flags.last_token.type == TOKEN.END_EXPR:
        self.set_mode(MODE.BlockStatement)
        self._flags.in_case_statement = True
    elif self._flags.case_body:
        self.set_mode(MODE.BlockStatement)
    elif second_token is not None and (second_token.text in [':', ','] and next_token.type in [TOKEN.STRING, TOKEN.WORD, TOKEN.RESERVED] or (next_token.text in ['get', 'set', '...'] and second_token.type in [TOKEN.WORD, TOKEN.RESERVED])):
        if self._last_last_text in ['class', 'interface'] and second_token.text not in [':', ',']:
            self.set_mode(MODE.BlockStatement)
        else:
            self.set_mode(MODE.ObjectLiteral)
    elif self._flags.last_token.type == TOKEN.OPERATOR and self._flags.last_token.text == '=>':
        self.set_mode(MODE.BlockStatement)
    elif self._flags.last_token.type in [TOKEN.EQUALS, TOKEN.START_EXPR, TOKEN.COMMA, TOKEN.OPERATOR] or reserved_array(self._flags.last_token, ['return', 'throw', 'import', 'default']):
        self.set_mode(MODE.ObjectLiteral)
    else:
        self.set_mode(MODE.BlockStatement)
    if self._flags.last_token:
        if reserved_array(self._flags.last_token.previous, ['class', 'extends']):
            self._flags.class_start_block = True
    empty_braces = next_token is not None and next_token.comments_before is None and (next_token.text == '}')
    empty_anonymous_function = empty_braces and self._flags.last_word == 'function' and (self._flags.last_token.type == TOKEN.END_EXPR)
    if self._options.brace_preserve_inline:
        index = 0
        check_token = None
        self._flags.inline_frame = True
        do_loop = True
        while do_loop:
            index += 1
            check_token = self._tokens.peek(index - 1)
            if check_token.newlines:
                self._flags.inline_frame = False
            do_loop = check_token.type != TOKEN.EOF and (not (check_token.type == TOKEN.END_BLOCK and check_token.opened == current_token))
    if (self._options.brace_style == 'expand' or (self._options.brace_style == 'none' and current_token.newlines)) and (not self._flags.inline_frame):
        if self._flags.last_token.type != TOKEN.OPERATOR and (empty_anonymous_function or self._flags.last_token.type == TOKEN.EQUALS or (reserved_array(self._flags.last_token, _special_word_set) and self._flags.last_token.text != 'else')):
            self._output.space_before_token = True
        else:
            self.print_newline(preserve_statement_flags=True)
    elif self.is_array(self._previous_flags.mode) and (self._flags.last_token.type == TOKEN.START_EXPR or self._flags.last_token.type == TOKEN.COMMA):
        if self._flags.inline_frame:
            self.allow_wrap_or_preserved_newline(current_token)
            self._flags.inline_frame = True
            self._previous_flags.multiline_frame = self._previous_flags.multiline_frame or self._flags.multiline_frame
            self._flags.multiline_frame = False
        elif self._flags.last_token.type == TOKEN.COMMA:
            self._output.space_before_token = True
    elif self._flags.last_token.type not in [TOKEN.OPERATOR, TOKEN.START_EXPR]:
        if self._flags.last_token.type in [TOKEN.START_BLOCK, TOKEN.SEMICOLON] and (not self._flags.inline_frame):
            self.print_newline()
        else:
            self._output.space_before_token = True
    self.print_token(current_token)
    self.indent()
    if not empty_braces and (not (self._options.brace_preserve_inline and self._flags.inline_frame)):
        self.print_newline()
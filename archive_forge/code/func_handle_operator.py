import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def handle_operator(self, current_token):
    isGeneratorAsterisk = current_token.text == '*' and (reserved_array(self._flags.last_token, ['function', 'yield']) or self._flags.last_token.type in [TOKEN.START_BLOCK, TOKEN.COMMA, TOKEN.END_BLOCK, TOKEN.SEMICOLON])
    isUnary = current_token.text in ['+', '-'] and (self._flags.last_token.type in [TOKEN.START_BLOCK, TOKEN.START_EXPR, TOKEN.EQUALS, TOKEN.OPERATOR] or self._flags.last_token.text in Tokenizer.line_starters or self._flags.last_token.text == ',')
    if self.start_of_statement(current_token):
        pass
    else:
        preserve_statement_flags = not isGeneratorAsterisk
        self.handle_whitespace_and_comments(current_token, preserve_statement_flags)
    if current_token.text == '*' and self._flags.last_token.type == TOKEN.DOT:
        self.print_token(current_token)
        return
    if current_token.text == '::':
        self.print_token(current_token)
        return
    if current_token.text in ['-', '+'] and self.start_of_object_property():
        self.print_token(current_token)
        return
    if self._flags.last_token.type == TOKEN.OPERATOR and self._options.operator_position in OPERATOR_POSITION_BEFORE_OR_PRESERVE:
        self.allow_wrap_or_preserved_newline(current_token)
    if current_token.text == ':' and self._flags.in_case:
        self.print_token(current_token)
        self._flags.in_case = False
        self._flags.case_body = True
        if self._tokens.peek().type != TOKEN.START_BLOCK:
            self.indent()
            self.print_newline()
            self._flags.case_block = False
        else:
            self._output.space_before_token = True
            self._flags.case_block = True
        return
    space_before = True
    space_after = True
    in_ternary = False
    if current_token.text == ':':
        if self._flags.ternary_depth == 0:
            space_before = False
        else:
            self._flags.ternary_depth -= 1
            in_ternary = True
    elif current_token.text == '?':
        self._flags.ternary_depth += 1
    if not isUnary and (not isGeneratorAsterisk) and self._options.preserve_newlines and (current_token.text in Tokenizer.positionable_operators):
        isColon = current_token.text == ':'
        isTernaryColon = isColon and in_ternary
        isOtherColon = isColon and (not in_ternary)
        if self._options.operator_position == OPERATOR_POSITION['before_newline']:
            self._output.space_before_token = not isOtherColon
            self.print_token(current_token)
            if not isColon or isTernaryColon:
                self.allow_wrap_or_preserved_newline(current_token)
            self._output.space_before_token = True
            return
        elif self._options.operator_position == OPERATOR_POSITION['after_newline']:
            self._output.space_before_token = True
            if not isColon or isTernaryColon:
                if self._tokens.peek().newlines:
                    self.print_newline(preserve_statement_flags=True)
                else:
                    self.allow_wrap_or_preserved_newline(current_token)
            else:
                self._output.space_before_token = False
            self.print_token(current_token)
            self._output.space_before_token = True
            return
        elif self._options.operator_position == OPERATOR_POSITION['preserve_newline']:
            if not isOtherColon:
                self.allow_wrap_or_preserved_newline(current_token)
            self._output.space_before_token = not (self._output.just_added_newline() or isOtherColon)
            self.print_token(current_token)
            self._output.space_before_token = True
            return
    if isGeneratorAsterisk:
        self.allow_wrap_or_preserved_newline(current_token)
        space_before = False
        next_token = self._tokens.peek()
        space_after = next_token and next_token.type in [TOKEN.WORD, TOKEN.RESERVED]
    elif current_token.text == '...':
        self.allow_wrap_or_preserved_newline(current_token)
        space_before = self._flags.last_token.type == TOKEN.START_BLOCK
        space_after = False
    elif current_token.text in ['--', '++', '!', '~'] or isUnary:
        if self._flags.last_token.type == TOKEN.COMMA or self._flags.last_token.type == TOKEN.START_EXPR:
            self.allow_wrap_or_preserved_newline(current_token)
        space_before = False
        space_after = False
        if current_token.newlines and (current_token.text == '--' or current_token.text == '++' or current_token.text == '~'):
            new_line_needed = reserved_array(self._flags.last_token, _special_word_set) and current_token.newlines
            if new_line_needed and (self._previous_flags.if_block or self._previous_flags.else_block):
                self.restore_mode()
            self.print_newline(new_line_needed, True)
        if self._flags.last_token.text == ';' and self.is_expression(self._flags.mode):
            space_before = True
        if self._flags.last_token.type == TOKEN.RESERVED:
            space_before = True
        elif self._flags.last_token.type == TOKEN.END_EXPR:
            space_before = not (self._flags.last_token.text == ']' and current_token.text in ['--', '++'])
        elif self._flags.last_token.type == TOKEN.OPERATOR:
            space_before = current_token.text in ['--', '-', '++', '+'] and self._flags.last_token.text in ['--', '-', '++', '+']
            if current_token.text in ['-', '+'] and self._flags.last_token.text in ['--', '++']:
                space_after = True
        if (self._flags.mode == MODE.BlockStatement and (not self._flags.inline_frame) or self._flags.mode == MODE.Statement) and self._flags.last_token.text in ['{', ';']:
            self.print_newline()
    if space_before:
        self._output.space_before_token = True
    self.print_token(current_token)
    if space_after:
        self._output.space_before_token = True
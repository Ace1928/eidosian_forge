import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def print_token_line_indentation(self, current_token):
    if self._output.just_added_newline():
        line = self._output.current_line
        if self._options.keep_array_indentation and current_token.newlines and (self.is_array(self._flags.mode) or current_token.text == '['):
            line.set_indent(-1)
            line.push(current_token.whitespace_before)
            self._output.space_before_token = False
        elif self._output.set_indent(self._flags.indentation_level, self._flags.alignment):
            self._flags.line_indent_level = self._flags.indentation_level
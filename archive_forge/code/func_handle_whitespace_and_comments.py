import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def handle_whitespace_and_comments(self, current_token, preserve_statement_flags=False):
    newlines = current_token.newlines
    keep_whitespace = self._options.keep_array_indentation and self.is_array(self._flags.mode)
    if current_token.comments_before is not None:
        for comment_token in current_token.comments_before:
            self.handle_whitespace_and_comments(comment_token, preserve_statement_flags)
            self.handle_token(comment_token, preserve_statement_flags)
    if keep_whitespace:
        for i in range(newlines):
            self.print_newline(i > 0, preserve_statement_flags)
    else:
        if self._options.max_preserve_newlines != 0 and newlines > self._options.max_preserve_newlines:
            newlines = self._options.max_preserve_newlines
        if self._options.preserve_newlines and newlines > 1:
            self.print_newline(False, preserve_statement_flags)
            for i in range(1, newlines):
                self.print_newline(True, preserve_statement_flags)
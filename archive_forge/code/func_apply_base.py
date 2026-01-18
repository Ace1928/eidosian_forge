import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def apply_base(self, flags_base, added_newline):
    next_indent_level = flags_base.indentation_level
    if not added_newline and flags_base.line_indent_level > next_indent_level:
        next_indent_level = flags_base.line_indent_level
    self.parent = flags_base
    self.last_token = flags_base.last_token
    self.last_word = flags_base.last_word
    self.indentation_level = next_indent_level
import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def restore_mode(self):
    if len(self._flag_store) > 0:
        self._previous_flags = self._flags
        self._flags = self._flag_store.pop()
        if self._previous_flags.mode == MODE.Statement:
            remove_redundant_indentation(self._output, self._previous_flags)
    self._output.set_indent(self._flags.indentation_level, self._flags.alignment)
import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def beautify(self, source_text='', opts=None):
    if opts is not None:
        self._options = BeautifierOptions(opts)
    source_text = source_text or ''
    if self._options.disabled:
        return source_text
    source_text = self._blank_state(source_text)
    source_text = self.unpack(source_text, self._options.eval_code)
    self._tokens = Tokenizer(source_text, self._options).tokenize()
    for current_token in self._tokens:
        self.handle_token(current_token)
        self._last_last_text = self._flags.last_token.text
        self._flags.last_token = current_token
    sweet_code = self._output.get_code(self._options.eol)
    return sweet_code
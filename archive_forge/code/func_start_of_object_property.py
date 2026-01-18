import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def start_of_object_property(self):
    return self._flags.parent.mode == MODE.ObjectLiteral and self._flags.mode == MODE.Statement and (self._flags.last_token.text == ':' and self._flags.ternary_depth == 0 or reserved_array(self._flags.last_token, ['get', 'set']))
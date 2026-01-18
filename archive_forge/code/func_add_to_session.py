import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def add_to_session(self, word):
    """Add a word to the session personal list."""
    self._check_this()
    _e.dict_add_to_session(self._this, word.encode())
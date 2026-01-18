import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def store_replacement(self, mis, cor):
    """Store a replacement spelling for a miss-spelled word.

        This method makes a suggestion to the spellchecking engine that the
        miss-spelled word <mis> is in fact correctly spelled as <cor>.  Such
        a suggestion will typically mean that <cor> appears early in the
        list of suggested spellings offered for later instances of <mis>.
        """
    if not mis:
        raise ValueError("can't store replacement for an empty string")
    if not cor:
        raise ValueError("can't store empty string as a replacement")
    self._check_this()
    _e.dict_store_replacement(self._this, mis.encode(), cor.encode())
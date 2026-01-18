import array
import warnings
import enchant
from enchant.errors import (
from enchant.tokenize import get_tokenizer
from enchant.utils import get_default_language
def add_to_personal(self, word=None):
    """Add given word to the personal word list.

        If no word is given, the current erroneous word is added.
        """
    warnings.warn('SpellChecker.add_to_personal is deprecated, please use SpellChecker.add', category=DeprecationWarning, stacklevel=2)
    self.add(word)
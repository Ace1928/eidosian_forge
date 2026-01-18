import array
import warnings
import enchant
from enchant.errors import (
from enchant.tokenize import get_tokenizer
from enchant.utils import get_default_language
def leading_context(self, chars):
    """Get <chars> characters of leading context.

        This method returns up to <chars> characters of leading
        context - the text that occurs in the string immediately
        before the current erroneous word.
        """
    start = max(self.wordpos - chars, 0)
    context = self._text[start:self.wordpos]
    return self._array_to_string(context)
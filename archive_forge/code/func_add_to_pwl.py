import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def add_to_pwl(self, word):
    """Add a word to the associated personal word list.

        This method adds the given word to the personal word list, and
        automatically saves the list to disk.
        """
    self._check_this()
    self.pwl.add_to_pwl(word)
    self.pel.remove(word)
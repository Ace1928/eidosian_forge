import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
@property
def stripped_strings(self):
    """Yield all strings in this PageElement, stripping them first.

        :yield: A sequence of stripped strings.
        """
    for string in self._all_strings(True):
        yield string
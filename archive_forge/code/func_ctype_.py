import os
import re
import sys
from typing import Any, Dict, List
from sphinx.errors import ExtensionError, SphinxError
from sphinx.search import SearchLanguage
from sphinx.util import import_object
def ctype_(self, char: str) -> str:
    for pattern, value in self.patterns_.items():
        if pattern.match(char):
            return value
    return 'O'
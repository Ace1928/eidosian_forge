from __future__ import unicode_literals
import re
from pybtex.exceptions import PybtexError
from pybtex import py3compat
def skip_to(self, patterns):
    end = None
    winning_pattern = None
    for pattern in patterns:
        match = pattern.search(self.text, self.pos)
        if match and (not end or match.end() < end):
            end = match.end()
            winning_pattern = pattern
    if winning_pattern:
        value = self.text[self.pos:end]
        self.pos = end
        self.update_lineno(value)
        return Token(value, winning_pattern)
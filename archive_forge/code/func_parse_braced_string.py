from __future__ import unicode_literals
import re
from pybtex.bibtex.utils import bibtex_abbreviate, bibtex_len
from pybtex.database import Person
from pybtex.scanner import (
def parse_braced_string(self):
    while True:
        try:
            token = self.required([self.TEXT, self.RBRACE, self.LBRACE])
        except PrematureEOF:
            raise UnbalancedBraceError(self)
        if token.pattern is self.TEXT:
            yield token.value
        elif token.pattern is self.RBRACE:
            break
        elif token.pattern is self.LBRACE:
            yield u'{{{0}}}'.format(''.join(self.parse_braced_string()))
        else:
            raise ValueError(token)
from __future__ import unicode_literals
import re
from pybtex.bibtex.utils import bibtex_abbreviate, bibtex_len
from pybtex.database import Person
from pybtex.scanner import (
def parse_toplevel(self):
    token = self.required([self.TEXT, self.LBRACE, self.RBRACE], allow_eof=True)
    if token.pattern is self.TEXT:
        return Text(token.value)
    elif token.pattern is self.LBRACE:
        return NamePart(self.parse_name_part())
    elif token.pattern is self.RBRACE:
        raise UnbalancedBraceError(self)
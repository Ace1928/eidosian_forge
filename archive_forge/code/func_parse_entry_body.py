from __future__ import unicode_literals
import re
from string import ascii_letters, digits
import six
from pybtex import textutils
from pybtex.bibtex.utils import split_name_list
from pybtex.database import Entry, Person, BibliographyDataError
from pybtex.database.input import BaseParser
from pybtex.scanner import (
from pybtex.utils import CaseInsensitiveDict, CaseInsensitiveSet
def parse_entry_body(self, body_end):
    if not self.keyless_entries:
        key_pattern = self.KEY_PAREN if body_end == self.RPAREN else self.KEY_BRACE
        self.current_entry_key = self.required([key_pattern]).value
    self.parse_entry_fields()
    if not self.want_current_entry():
        raise SkipEntry
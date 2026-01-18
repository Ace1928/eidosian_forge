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
def substitute_macro(self, name):
    try:
        return self.macros[name]
    except KeyError:
        if self.want_current_entry():
            self.handle_error(UndefinedMacro(name, self))
        return ''
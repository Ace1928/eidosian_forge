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
def process_preamble(self, value_list):
    value = textutils.normalize_whitespace(self.flatten_value_list(value_list))
    self.data.add_to_preamble(value)
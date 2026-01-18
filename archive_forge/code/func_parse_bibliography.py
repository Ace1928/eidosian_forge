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
def parse_bibliography(self):
    while True:
        if not self.skip_to([self.AT]):
            return
        self.command_start = self.pos - 1
        try:
            yield tuple(self.parse_command())
        except PybtexSyntaxError as error:
            self.handle_error(error)
        except SkipEntry:
            pass
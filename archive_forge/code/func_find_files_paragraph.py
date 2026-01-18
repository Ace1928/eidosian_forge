import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
def find_files_paragraph(self, filename):
    """Returns the FilesParagraph for the given filename.

        In accordance with the spec, this method returns the last FilesParagraph
        that matches the filename.  If no paragraphs matched, returns None.
        """
    result = None
    for p in self.all_files_paragraphs():
        if p.matches(filename):
            result = p
    return result
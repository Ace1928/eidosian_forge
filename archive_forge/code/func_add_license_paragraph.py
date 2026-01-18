import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
def add_license_paragraph(self, paragraph):
    """Adds a LicenceParagraph to this object.

        The paragraph is inserted after any other paragraphs.
        """
    if not isinstance(paragraph, LicenseParagraph):
        raise TypeError('paragraph must be a LicenseParagraph instance')
    self.__paragraphs.append(paragraph)
    self.__file.append(paragraph._underlying_paragraph)
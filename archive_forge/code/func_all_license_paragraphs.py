import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
def all_license_paragraphs(self):
    """Returns an iterator over standalone LicenseParagraph objects."""
    return (p for p in self.__paragraphs if isinstance(p, LicenseParagraph))
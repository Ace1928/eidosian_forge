import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
def current_format(self):
    """Returns True iff the format is the current format."""
    return self.format == _CURRENT_FORMAT
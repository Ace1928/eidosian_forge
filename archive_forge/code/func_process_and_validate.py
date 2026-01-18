import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
def process_and_validate(s):
    s = s.strip()
    if not s:
        raise MachineReadableFormatError('values must not be empty')
    if '\n' in s:
        raise MachineReadableFormatError('values must not contain newlines')
    return s
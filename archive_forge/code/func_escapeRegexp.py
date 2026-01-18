from __future__ import absolute_import, division, unicode_literals
import re
import warnings
from .constants import DataLossWarning
def escapeRegexp(string):
    specialCharacters = ('.', '^', '$', '*', '+', '?', '{', '}', '[', ']', '|', '(', ')', '-')
    for char in specialCharacters:
        string = string.replace(char, '\\' + char)
    return string
import codecs
import re
import warnings
from typing import Match
def decodeStringEscape(s):
    warnings.warn(DeprecationWarning('rdflib.compat.decodeStringEscape() is deprecated, it will be removed in rdflib 7.0.0. This function is not used anywhere in rdflib anymore and the utility that it does provide is not implemented correctly.'))
    '\n    s is byte-string - replace \\ escapes in string\n    '
    s = s.replace('\\t', '\t')
    s = s.replace('\\n', '\n')
    s = s.replace('\\r', '\r')
    s = s.replace('\\b', '\x08')
    s = s.replace('\\f', '\x0c')
    s = s.replace('\\"', '"')
    s = s.replace("\\'", "'")
    s = s.replace('\\\\', '\\')
    return s
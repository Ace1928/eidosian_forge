from . import idnadata
import bisect
import unicodedata
import re
from typing import Union, Optional
from .intranges import intranges_contain
def check_hyphen_ok(label: str) -> bool:
    if label[2:4] == '--':
        raise IDNAError('Label has disallowed hyphens in 3rd and 4th position')
    if label[0] == '-' or label[-1] == '-':
        raise IDNAError('Label must not start or end with a hyphen')
    return True
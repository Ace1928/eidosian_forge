import warnings
from collections import Counter
from encodings.aliases import aliases
from hashlib import sha256
from json import dumps
from re import sub
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from .constant import NOT_PRINTABLE_PATTERN, TOO_BIG_SEQUENCE
from .md import mess_ratio
from .utils import iana_name, is_multi_byte_encoding, unicode_range
def add_submatch(self, other: 'CharsetMatch') -> None:
    if not isinstance(other, CharsetMatch) or other == self:
        raise ValueError('Unable to add instance <{}> as a submatch of a CharsetMatch'.format(other.__class__))
    other._string = None
    self._leaves.append(other)
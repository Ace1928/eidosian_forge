import codecs
import itertools
import sys
from enum import Enum, auto
from typing import Optional, List, Sequence, Union
from .termhelpers import Termmode
from .curtsieskeys import CURTSIES_NAMES as special_curtsies_names
def decodable(seq: bytes, encoding: str) -> bool:
    try:
        u = seq.decode(encoding)
    except UnicodeDecodeError:
        return False
    else:
        return True
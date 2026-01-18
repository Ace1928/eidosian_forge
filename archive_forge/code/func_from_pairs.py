import re
from .. import osutils
from ..iterablefile import IterableFile
@classmethod
def from_pairs(cls, pairs):
    ret = cls()
    ret.items = pairs
    return ret
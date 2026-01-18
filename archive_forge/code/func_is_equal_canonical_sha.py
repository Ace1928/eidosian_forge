import zlib
from gitdb.util import byte_ord
import mmap
from itertools import islice
from functools import reduce
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_text
from gitdb.typ import (
from io import StringIO
def is_equal_canonical_sha(canonical_length, match, sha1):
    """
    :return: True if the given lhs and rhs 20 byte binary shas
        The comparison will take the canonical_length of the match sha into account,
        hence the comparison will only use the last 4 bytes for uneven canonical representations
    :param match: less than 20 byte sha
    :param sha1: 20 byte sha"""
    binary_length = canonical_length // 2
    if match[:binary_length] != sha1[:binary_length]:
        return False
    if canonical_length - binary_length and (byte_ord(match[-1]) ^ byte_ord(sha1[len(match) - 1])) & 240:
        return False
    return True
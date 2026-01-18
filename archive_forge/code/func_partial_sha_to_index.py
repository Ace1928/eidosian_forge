import zlib
from gitdb.exc import (
from gitdb.util import (
from gitdb.fun import (
from gitdb.base import (      # Amazing !
from gitdb.stream import (
from struct import pack
from binascii import crc32
from gitdb.const import NULL_BYTE
import tempfile
import array
import os
import sys
def partial_sha_to_index(self, partial_bin_sha, canonical_length):
    """
        :return: index as in `sha_to_index` or None if the sha was not found in this
            index file
        :param partial_bin_sha: an at least two bytes of a partial binary sha as bytes
        :param canonical_length: length of the original hexadecimal representation of the
            given partial binary sha
        :raise AmbiguousObjectName:"""
    if len(partial_bin_sha) < 2:
        raise ValueError('Require at least 2 bytes of partial sha')
    assert isinstance(partial_bin_sha, bytes), 'partial_bin_sha must be bytes'
    first_byte = byte_ord(partial_bin_sha[0])
    get_sha = self.sha
    lo = 0
    if first_byte != 0:
        lo = self._fanout_table[first_byte - 1]
    hi = self._fanout_table[first_byte]
    filled_sha = partial_bin_sha + NULL_BYTE * (20 - len(partial_bin_sha))
    while lo < hi:
        mid = (lo + hi) // 2
        mid_sha = get_sha(mid)
        if filled_sha < mid_sha:
            hi = mid
        elif filled_sha == mid_sha:
            lo = mid
            break
        else:
            lo = mid + 1
    if lo < self.size():
        cur_sha = get_sha(lo)
        if is_equal_canonical_sha(canonical_length, partial_bin_sha, cur_sha):
            next_sha = None
            if lo + 1 < self.size():
                next_sha = get_sha(lo + 1)
            if next_sha and next_sha == cur_sha:
                raise AmbiguousObjectName(partial_bin_sha)
            return lo
    return None
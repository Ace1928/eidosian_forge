import binascii
import os
import struct
from .dirstate import DirState, DirstateCorrupt
def bisect_dirblock(dirblocks, dirname, lo=0, hi=None, cache={}):
    """Return the index where to insert dirname into the dirblocks.

    The return value idx is such that all directories blocks in dirblock[:idx]
    have names < dirname, and all blocks in dirblock[idx:] have names >=
    dirname.

    Optional args lo (default 0) and hi (default len(dirblocks)) bound the
    slice of a to be searched.
    """
    if hi is None:
        hi = len(dirblocks)
    try:
        dirname_split = cache[dirname]
    except KeyError:
        dirname_split = dirname.split(b'/')
        cache[dirname] = dirname_split
    while lo < hi:
        mid = (lo + hi) // 2
        cur = dirblocks[mid][0]
        try:
            cur_split = cache[cur]
        except KeyError:
            cur_split = cur.split(b'/')
            cache[cur] = cur_split
        if cur_split < dirname_split:
            lo = mid + 1
        else:
            hi = mid
    return lo
from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
def read_segments(self, offsets: Sequence[int], bytecounts: Sequence[int], /, indices: Sequence[int] | None=None, *, sort: bool=True, lock: threading.RLock | NullContext | None=None, buffersize: int | None=None, flat: bool=True) -> Iterator[tuple[bytes | None, int]] | Iterator[list[tuple[bytes | None, int]]]:
    """Return iterator over segments read from file and their indices.

        The purpose of this function is to

        - reduce small or random reads.
        - reduce acquiring reentrant locks.
        - synchronize seeks and reads.
        - limit size of segments read into memory at once.
          (ThreadPoolExecutor.map is not collecting iterables lazily).

        Parameters:
            offsets:
                Offsets of segments to read from file.
            bytecounts:
                Byte counts of segments to read from file.
            indices:
                Indices of segments in image.
                The default is `range(len(offsets))`.
            sort:
                Read segments from file in order of their offsets.
            lock:
                Reentrant lock to synchronize seeks and reads.
            buffersize:
                Approximate number of bytes to read from file in one pass.
                The default is :py:attr:`_TIFF.BUFFERSIZE`.
            flat:
                If *True*, return iterator over individual (segment, index)
                tuples.
                Else, return an iterator over a list of (segment, index)
                tuples that were acquired in one pass.

        Yields:
            Individual or lists of `(segment, index)` tuples.

        """
    assert self._fh is not None
    length = len(offsets)
    if length < 1:
        return
    if length == 1:
        index = 0 if indices is None else indices[0]
        if bytecounts[index] > 0 and offsets[index] > 0:
            if lock is None:
                lock = self._lock
            with lock:
                self.seek(offsets[index])
                data = self._fh.read(bytecounts[index])
        else:
            data = None
        yield ((data, index) if flat else [(data, index)])
        return
    if lock is None:
        lock = self._lock
    if buffersize is None:
        buffersize = TIFF.BUFFERSIZE
    if indices is None:
        segments = [(i, offsets[i], bytecounts[i]) for i in range(length)]
    else:
        segments = [(indices[i], offsets[i], bytecounts[i]) for i in range(length)]
    if sort:
        segments = sorted(segments, key=lambda x: x[1])
    iscontig = True
    for i in range(length - 1):
        _, offset, bytecount = segments[i]
        nextoffset = segments[i + 1][1]
        if offset == 0 or bytecount == 0 or nextoffset == 0:
            continue
        if offset + bytecount != nextoffset:
            iscontig = False
            break
    seek = self.seek
    read = self._fh.read
    result: list[tuple[bytes | None, int]]
    if iscontig:
        i = 0
        while i < length:
            j = i
            offset = -1
            bytecount = 0
            while bytecount <= buffersize and i < length:
                _, o, b = segments[i]
                if o > 0 and b > 0:
                    if offset < 0:
                        offset = o
                    bytecount += b
                i += 1
            if offset < 0:
                data = None
            else:
                with lock:
                    seek(offset)
                    data = read(bytecount)
            start = 0
            stop = 0
            result = []
            while j < i:
                index, offset, bytecount = segments[j]
                if offset > 0 and bytecount > 0:
                    stop += bytecount
                    result.append((data[start:stop], index))
                    start = stop
                else:
                    result.append((None, index))
                j += 1
            if flat:
                yield from result
            else:
                yield result
        return
    i = 0
    while i < length:
        result = []
        size = 0
        with lock:
            while size <= buffersize and i < length:
                index, offset, bytecount = segments[i]
                if offset > 0 and bytecount > 0:
                    seek(offset)
                    result.append((read(bytecount), index))
                    size += bytecount
                else:
                    result.append((None, index))
                i += 1
        if flat:
            yield from result
        else:
            yield result
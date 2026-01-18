from collections import UserDict, deque
from functools import partial
from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
from . import DefaultTable
import array
import itertools
import logging
import struct
import sys
import fontTools.ttLib.tables.TupleVariation as tv
@staticmethod
def compileOffsets_(offsets):
    """Packs a list of offsets into a 'gvar' offset table.

        Returns a pair (bytestring, tableFormat). Bytestring is the
        packed offset table. Format indicates whether the table
        uses short (tableFormat=0) or long (tableFormat=1) integers.
        The returned tableFormat should get packed into the flags field
        of the 'gvar' header.
        """
    assert len(offsets) >= 2
    for i in range(1, len(offsets)):
        assert offsets[i - 1] <= offsets[i]
    if max(offsets) <= 65535 * 2:
        packed = array.array('H', [n >> 1 for n in offsets])
        tableFormat = 0
    else:
        packed = array.array('I', offsets)
        tableFormat = 1
    if sys.byteorder != 'big':
        packed.byteswap()
    return (packed.tobytes(), tableFormat)
from fontTools.config import OPTIONS
from fontTools.misc.textTools import Tag, bytesjoin
from .DefaultTable import DefaultTable
from enum import IntEnum
import sys
import array
import struct
import logging
from functools import lru_cache
from typing import Iterator, NamedTuple, Optional, Tuple
def getAllDataUsingHarfbuzz(self, tableTag):
    """The Whole table is represented as a Graph.
        Assemble graph data and call Harfbuzz repacker to pack the table.
        Harfbuzz repacker is faster and retain as much sub-table sharing as possible, see also:
        https://github.com/harfbuzz/harfbuzz/blob/main/docs/repacker.md
        The input format for hb.repack() method is explained here:
        https://github.com/harfbuzz/uharfbuzz/blob/main/src/uharfbuzz/_harfbuzz.pyx#L1149
        """
    internedTables = {}
    self._doneWriting(internedTables, shareExtension=True)
    tables = []
    obj_list = []
    done = {}
    objidx = 0
    virtual_edges = []
    self._gatherGraphForHarfbuzz(tables, obj_list, done, objidx, virtual_edges)
    pos = 0
    for table in tables:
        table.pos = pos
        pos = pos + table.getDataLength()
    data = []
    for table in tables:
        tableData = table.getDataForHarfbuzz()
        data.append(tableData)
    if hasattr(hb, 'repack_with_tag'):
        return hb.repack_with_tag(str(tableTag), data, obj_list)
    else:
        return hb.repack(data, obj_list)
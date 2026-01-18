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
def getAllData(self, remove_duplicate=True):
    """Assemble all data, including all subtables."""
    if remove_duplicate:
        internedTables = {}
        self._doneWriting(internedTables)
    tables = []
    extTables = []
    done = {}
    self._gatherTables(tables, extTables, done)
    tables.reverse()
    extTables.reverse()
    pos = 0
    for table in tables:
        table.pos = pos
        pos = pos + table.getDataLength()
    for table in extTables:
        table.pos = pos
        pos = pos + table.getDataLength()
    data = []
    for table in tables:
        tableData = table.getData()
        data.append(tableData)
    for table in extTables:
        tableData = table.getData()
        data.append(tableData)
    return bytesjoin(data)
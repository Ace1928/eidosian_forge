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
def iterSubTables(self) -> Iterator[SubTableEntry]:
    """Yield (name, value, index) namedtuples for all subtables of current table.

        A sub-table is an instance of BaseTable (or subclass thereof) that is a child
        of self, the current parent table.
        The tuples also contain the attribute name (str) of the of parent table to get
        a subtable, and optionally, for lists of subtables (i.e. attributes associated
        with a converter that has a 'repeat'), an index into the list containing the
        given subtable value.
        This method can be useful to traverse trees of otTables.
        """
    for conv in self.getConverters():
        name = conv.name
        value = getattr(self, name, None)
        if value is None:
            continue
        if isinstance(value, BaseTable):
            yield self.SubTableEntry(name, value)
        elif isinstance(value, list):
            yield from (self.SubTableEntry(name, v, index=i) for i, v in enumerate(value) if isinstance(v, BaseTable))
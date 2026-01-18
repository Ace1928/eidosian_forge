from __future__ import division, print_function, absolute_import
from petl.compat import next, string_types
from petl.util.base import iterpeek, ValuesView, Table
from petl.util.materialise import columns
def infer_dtype(table):
    import numpy as np
    it = iter(table)
    hdr = next(it)
    flds = list(map(str, hdr))
    rows = tuple(it)
    dtype = np.rec.array(rows).dtype
    dtype.names = flds
    return dtype
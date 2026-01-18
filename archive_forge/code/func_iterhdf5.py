from __future__ import absolute_import, print_function, division
from contextlib import contextmanager
from petl.compat import string_types
from petl.errors import ArgumentError
from petl.util.base import Table, iterpeek, data
from petl.io.numpy import infer_dtype
def iterhdf5(source, where, name, condition, condvars, start, stop, step):
    with _get_hdf5_table(source, where, name) as h5tbl:
        hdr = tuple(h5tbl.colnames)
        yield hdr
        if condition is not None:
            it = h5tbl.where(condition, condvars=condvars, start=start, stop=stop, step=step)
        else:
            it = h5tbl.iterrows(start=start, stop=stop, step=step)
        for row in it:
            yield row[:]
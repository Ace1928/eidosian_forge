from __future__ import absolute_import, print_function, division
from contextlib import contextmanager
from petl.compat import string_types
from petl.errors import ArgumentError
from petl.util.base import Table, iterpeek, data
from petl.io.numpy import infer_dtype
def appendhdf5(table, source, where=None, name=None):
    """
    As :func:`petl.io.hdf5.tohdf5` but don't truncate the target table before
    loading.

    """
    with _get_hdf5_table(source, where, name, mode='a') as h5table:
        _insert(table, h5table)
from numba.tests.support import TestCase, linux_only
from numba.tests.gdb_support import needs_gdb, skip_unless_pexpect, GdbMIDriver
from unittest.mock import patch, Mock
from numba.core import datamodel
import numpy as np
from numba import typeof
import ctypes as ct
import unittest
def get_gdb_repr(self, array):
    from numba.misc import gdb_print_extension

    class DISubrange:

        def __init__(self, lo, hi):
            self._lo = lo
            self._hi = hi

        @property
        def type(self):
            return self

        def range(self):
            return (self._lo, self._hi)

    class DW_TAG_array_type:

        def __init__(self, lo, hi):
            self._lo, self._hi = (lo, hi)

        def fields(self):
            return [DISubrange(self._lo, self._hi)]

    class DIDerivedType_tuple:

        def __init__(self, the_tuple):
            self._type = DW_TAG_array_type(0, len(the_tuple) - 1)
            self._tuple = the_tuple

        @property
        def type(self):
            return self._type

        def __getitem__(self, item):
            return self._tuple[item]

    class DICompositeType_Array:

        def __init__(self, arr, type_str):
            self._arr = arr
            self._type_str = type_str

        def __getitem__(self, item):
            return getattr(self, item)

        @property
        def data(self):
            return self._arr.ctypes.data

        @property
        def itemsize(self):
            return self._arr.itemsize

        @property
        def shape(self):
            return DIDerivedType_tuple(self._arr.shape)

        @property
        def strides(self):
            return DIDerivedType_tuple(self._arr.strides)

        @property
        def type(self):
            return self._type_str
    dmm = datamodel.default_manager
    array_model = datamodel.models.ArrayModel(dmm, typeof(array))
    data_type = array_model.get_data_type()
    type_str = f'{typeof(array)} ({data_type.structure_repr()})'
    fake_gdb_arr = DICompositeType_Array(array, type_str)
    printer = gdb_print_extension.NumbaArrayPrinter(fake_gdb_arr)
    return printer.to_string().strip()
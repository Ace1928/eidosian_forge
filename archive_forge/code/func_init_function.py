import sys
from unittest import mock
import types
import warnings
import unittest
import os
import subprocess
import threading
from numba import config, njit
from numba.tests.support import TestCase
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
def init_function():

    class DummyType(numba.types.Type):

        def __init__(self):
            super(DummyType, self).__init__(name='DummyType')

    @numba.extending.typeof_impl.register(_DummyClass)
    def typer_DummyClass(val, c):
        return DummyType()

    @numba.extending.register_model(DummyType)
    class DummyModel(numba.extending.models.StructModel):

        def __init__(self, dmm, fe_type):
            members = [('value', numba.types.float64)]
            super(DummyModel, self).__init__(dmm, fe_type, members)

    @numba.extending.unbox(DummyType)
    def unbox_dummy(typ, obj, c):
        value_obj = c.pyapi.object_getattr_string(obj, 'value')
        dummy_struct_proxy = numba.core.cgutils.create_struct_proxy(typ)
        dummy_struct = dummy_struct_proxy(c.context, c.builder)
        dummy_struct.value = c.pyapi.float_as_double(value_obj)
        c.pyapi.decref(value_obj)
        err_flag = c.pyapi.err_occurred()
        is_error = numba.core.cgutils.is_not_null(c.builder, err_flag)
        return numba.extending.NativeValue(dummy_struct._getvalue(), is_error=is_error)

    @numba.extending.box(DummyType)
    def box_dummy(typ, val, c):
        dummy_struct_proxy = numba.core.cgutils.create_struct_proxy(typ)
        dummy_struct = dummy_struct_proxy(c.context, c.builder)
        value_obj = c.pyapi.float_from_double(dummy_struct.value)
        serialized_clazz = c.pyapi.serialize_object(_DummyClass)
        class_obj = c.pyapi.unserialize(serialized_clazz)
        res = c.pyapi.call_function_objargs(class_obj, (value_obj,))
        c.pyapi.decref(value_obj)
        c.pyapi.decref(class_obj)
        return res
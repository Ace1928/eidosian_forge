from collections import namedtuple
import gc
import os
import operator
import sys
import weakref
import numpy as np
from numba.core import types, typing, errors, sigutils
from numba.core.types.abstract import _typecache
from numba.core.types.functions import _header_lead
from numba.core.typing.templates import make_overload_template
from numba import jit, njit, typeof
from numba.core.extending import (overload, register_model, models, unbox,
from numba.tests.support import TestCase, create_temp_module
from numba.tests.enum_usecases import Color, Shake, Shape
import unittest
from numba.np import numpy_support
from numba.core import types
def test_externally_defined_type_is_external(self):
    with create_temp_module(self.source_lines) as test_module:
        FooType = test_module.FooType
        self.assertFalse(FooType().is_internal)

        class Foo(object):
            pass
        register_model(FooType)(models.OpaqueModel)

        @typeof_impl.register(Foo)
        def _typ_foo(val, c):
            return FooType()

        @unbox(FooType)
        def unbox_foo(typ, obj, c):
            return NativeValue(c.context.get_dummy_value())

        def false_if_not_array(a):
            pass

        @overload(false_if_not_array)
        def ol_false_if_not_array(a):
            if isinstance(a, types.Array):
                return lambda a: True
            else:
                return lambda a: False

        @njit
        def call_false_if_not_array(a):
            return false_if_not_array(a)
        self.assertTrue(call_false_if_not_array(np.zeros(10)))
        self.assertFalse(call_false_if_not_array(10))
        self.assertFalse(call_false_if_not_array(Foo()))

        def false_if_not_array_closed_system(a):
            pass

        @overload(false_if_not_array_closed_system)
        def ol_false_if_not_array_closed_system(a):
            if a.is_internal:
                if isinstance(a, types.Array):
                    return lambda a: True
                else:
                    return lambda a: False

        @njit
        def call_false_if_not_array_closed_system(a):
            return false_if_not_array_closed_system(a)
        self.assertTrue(call_false_if_not_array_closed_system(np.zeros(10)))
        self.assertFalse(call_false_if_not_array_closed_system(10))
        with self.assertRaises(errors.TypingError) as raises:
            call_false_if_not_array_closed_system(Foo())
        estr = str(raises.exception)
        self.assertIn(_header_lead, estr)
        self.assertIn('false_if_not_array_closed_system', estr)
        self.assertIn('(Foo)', estr)
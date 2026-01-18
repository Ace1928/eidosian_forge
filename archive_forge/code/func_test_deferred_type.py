import ctypes
import itertools
import pickle
import random
import typing as pt
import unittest
from collections import OrderedDict
import numpy as np
from numba import (boolean, deferred_type, float32, float64, int16, int32,
from numba.core import errors, types
from numba.core.dispatcher import Dispatcher
from numba.core.errors import LoweringError, TypingError
from numba.core.runtime.nrt import MemInfo
from numba.experimental import jitclass
from numba.experimental.jitclass import _box
from numba.experimental.jitclass.base import JitClassType
from numba.tests.support import MemoryLeakMixin, TestCase, skip_if_typeguard
from numba.tests.support import skip_unless_scipy
def test_deferred_type(self):
    node_type = deferred_type()
    spec = OrderedDict()
    spec['data'] = float32
    spec['next'] = optional(node_type)

    @njit
    def get_data(node):
        return node.data

    @jitclass(spec)
    class LinkedNode(object):

        def __init__(self, data, next):
            self.data = data
            self.next = next

        def get_next_data(self):
            return get_data(self.next)

        def append_to_tail(self, other):
            cur = self
            while cur.next is not None:
                cur = cur.next
            cur.next = other
    node_type.define(LinkedNode.class_type.instance_type)
    first = LinkedNode(123, None)
    self.assertEqual(first.data, 123)
    self.assertIsNone(first.next)
    second = LinkedNode(321, first)
    first_meminfo = _get_meminfo(first)
    second_meminfo = _get_meminfo(second)
    self.assertEqual(first_meminfo.refcount, 3)
    self.assertEqual(second.next.data, first.data)
    self.assertEqual(first_meminfo.refcount, 3)
    self.assertEqual(second_meminfo.refcount, 2)
    first_val = second.get_next_data()
    self.assertEqual(first_val, first.data)
    self.assertIsNone(first.next)
    second.append_to_tail(LinkedNode(567, None))
    self.assertIsNotNone(first.next)
    self.assertEqual(first.next.data, 567)
    self.assertIsNone(first.next.next)
    second.append_to_tail(LinkedNode(678, None))
    self.assertIsNotNone(first.next.next)
    self.assertEqual(first.next.next.data, 678)
    self.assertEqual(first_meminfo.refcount, 3)
    del second, second_meminfo
    self.assertEqual(first_meminfo.refcount, 2)
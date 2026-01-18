import contextlib
import difflib
import pprint
import pickle
import re
import sys
import logging
import warnings
import weakref
import inspect
import types
from copy import deepcopy
from test import support
import unittest
from unittest.test.support import (
from test.support import captured_stderr, gc_collect
def testAssertEqual(self):
    equal_pairs = [((), ()), ({}, {}), ([], []), (set(), set()), (frozenset(), frozenset())]
    for a, b in equal_pairs:
        try:
            self.assertEqual(a, b)
        except self.failureException:
            self.fail('assertEqual(%r, %r) failed' % (a, b))
        try:
            self.assertEqual(a, b, msg='foo')
        except self.failureException:
            self.fail('assertEqual(%r, %r) with msg= failed' % (a, b))
        try:
            self.assertEqual(a, b, 'foo')
        except self.failureException:
            self.fail('assertEqual(%r, %r) with third parameter failed' % (a, b))
    unequal_pairs = [((), []), ({}, set()), (set([4, 1]), frozenset([4, 2])), (frozenset([4, 5]), set([2, 3])), (set([3, 4]), set([5, 4]))]
    for a, b in unequal_pairs:
        self.assertRaises(self.failureException, self.assertEqual, a, b)
        self.assertRaises(self.failureException, self.assertEqual, a, b, 'foo')
        self.assertRaises(self.failureException, self.assertEqual, a, b, msg='foo')
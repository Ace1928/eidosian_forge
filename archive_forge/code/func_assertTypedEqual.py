from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def assertTypedEqual(self, actual, expect, msg=None):
    self.assertEqual(actual, expect, msg)

    def recurse(actual, expect):
        if isinstance(expect, (tuple, list)):
            for x, y in zip(actual, expect):
                recurse(x, y)
        else:
            self.assertIs(type(actual), type(expect), msg)
    recurse(actual, expect)
from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_449964(self):
    self.assertEqual(regex.sub('(?P<unk>x)', '\\g<1>\\g<1>\\b', 'xx'), 'xx\x08xx\x08')
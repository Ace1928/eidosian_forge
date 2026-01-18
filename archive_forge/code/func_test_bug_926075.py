from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_926075(self):
    if regex.compile('bug_926075') is regex.compile(b'bug_926075'):
        self.fail()
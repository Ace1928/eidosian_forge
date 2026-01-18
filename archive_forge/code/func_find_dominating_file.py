from __future__ import absolute_import
from __future__ import print_function
import sys
import os
from unittest import TestCase as NonLeakingTestCase
import greenlet
def find_dominating_file(name):
    if os.path.exists(name):
        return name
    tried = []
    here = os.path.abspath(os.path.dirname(__file__))
    for i in range(10):
        up = ['..'] * i
        path = [here] + up + [name]
        fname = os.path.join(*path)
        fname = os.path.abspath(fname)
        tried.append(fname)
        if os.path.exists(fname):
            return fname
    raise AssertionError('Could not find file ' + name + '; checked ' + str(tried))
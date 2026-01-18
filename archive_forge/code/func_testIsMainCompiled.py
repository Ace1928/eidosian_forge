import datetime
import sys
import types
import unittest
import six
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testIsMainCompiled(self):
    module = self.CreateModule('__main__')
    module.__file__ = '/bing/blam/bloom/blarm/my_file.pyc'
    self.assertPackageEquals('my_file', util.get_package_for_module(module))
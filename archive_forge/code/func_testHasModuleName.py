import datetime
import sys
import types
import unittest
import six
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testHasModuleName(self):
    module = self.CreateModule('service_module')
    self.assertPackageEquals('service_module', util.get_package_for_module(module))
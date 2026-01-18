import datetime
import sys
import types
import unittest
import six
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testModuleNameNotInSys(self):
    self.assertPackageEquals(None, util.get_package_for_module('service_module'))
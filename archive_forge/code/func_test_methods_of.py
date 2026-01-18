import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
def test_methods_of(self):

    class DummyClass(object):

        def dummyMethod(self):
            print('just for test')
    obj = DummyClass()
    result = common.methods_of(obj)
    self.assertEqual(1, len(result))
    method = result['dummyMethod']
    self.assertIsNotNone(method)
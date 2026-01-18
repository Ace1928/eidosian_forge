import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testUndescribable(self):

    class NonService(object):

        def fn(self):
            pass
    for value in (NonService, NonService.fn, 1, 'string', 1.2, None):
        self.assertEquals(None, descriptor.describe(value))
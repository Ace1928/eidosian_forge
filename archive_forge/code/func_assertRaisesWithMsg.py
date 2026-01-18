import copy
import pickle
import sys
import tempfile
import six
import unittest2 as unittest
import mock
from mock import (
from mock.mock import _CallList
from mock.tests.support import (
def assertRaisesWithMsg(self, exception, message, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except:
        instance = sys.exc_info()[1]
        self.assertIsInstance(instance, exception)
    else:
        self.fail('Exception %r not raised' % (exception,))
    msg = str(instance)
    self.assertEqual(msg, message)
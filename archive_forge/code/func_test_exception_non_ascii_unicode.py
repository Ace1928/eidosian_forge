import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_exception_non_ascii_unicode(self):
    hi_ru = u'привет'
    fail = failure.Failure.from_exception(ValueError(hi_ru))
    self.assertEqual(hi_ru, fail.exception_str)
    self.assertIsInstance(fail.exception_str, str)
    self.assertEqual(u'Failure: ValueError: %s' % hi_ru, str(fail))
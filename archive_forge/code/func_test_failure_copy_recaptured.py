import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_failure_copy_recaptured(self):
    captured = _captured_failure('Woot!')
    fail_obj = failure.Failure(exception_str=captured.exception_str, traceback_str=captured.traceback_str, exc_type_names=list(captured))
    copied = fail_obj.copy()
    self.assertIsNot(fail_obj, copied)
    self.assertEqual(fail_obj, copied)
    self.assertFalse(fail_obj != copied)
    self.assertTrue(fail_obj.matches(copied))
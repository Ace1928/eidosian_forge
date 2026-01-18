import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_no_type_names(self):
    fail_obj = _captured_failure('Woot!')
    fail_obj = failure.Failure(exception_str=fail_obj.exception_str, traceback_str=fail_obj.traceback_str, exc_type_names=[])
    self.assertEqual([], list(fail_obj))
    self.assertEqual('Failure: Woot!', fail_obj.pformat())
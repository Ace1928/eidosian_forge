import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_check_str_not_there(self):
    val = 'ValueError'
    self.assertIsNone(self.fail_obj.check(val))
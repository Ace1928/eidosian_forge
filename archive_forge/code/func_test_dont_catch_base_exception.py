import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_dont_catch_base_exception(self):
    try:
        raise SystemExit()
    except BaseException:
        self.assertRaises(TypeError, failure.Failure)
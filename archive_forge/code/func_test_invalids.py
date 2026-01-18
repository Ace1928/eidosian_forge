import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_invalids(self):
    f = {'exception_str': 'blah', 'traceback_str': 'blah', 'exc_type_names': []}
    self.assertRaises(exceptions.InvalidFormat, failure.Failure.validate, f)
    f = {'exception_str': 'blah', 'exc_type_names': ['Exception']}
    self.assertRaises(exceptions.InvalidFormat, failure.Failure.validate, f)
    f = {'exception_str': 'blah', 'traceback_str': 'blah', 'exc_type_names': ['Exception'], 'version': -1}
    self.assertRaises(exceptions.InvalidFormat, failure.Failure.validate, f)
import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_empty_does_not_reraise(self):
    self.assertIsNone(failure.Failure.reraise_if_any([]))
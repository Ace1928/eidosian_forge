import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_compares_to_none(self):
    captured = _captured_failure('Woot!')
    self.assertIsNotNone(captured)
    self.assertFalse(captured.matches(None))
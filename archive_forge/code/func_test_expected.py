import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_expected(self):
    self.assertEqual(self.expected, misc.safe_copy_dict(self.original))
    self.assertFalse(self.expected is misc.safe_copy_dict(self.original))
import logging
import unittest
import weakref
from typing import Dict, List
from .. import pyutils
def fail_uncollected(self):
    self.fail('Uncollected test case: ' + self._problem_case_id)
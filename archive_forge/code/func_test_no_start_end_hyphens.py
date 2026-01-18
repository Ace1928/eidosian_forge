import re
import unittest
from oslo_config import types
def test_no_start_end_hyphens(self):
    self.assertInvalid('-host.com')
    self.assertInvalid('-hostname.com-')
    self.assertInvalid('hostname.co.uk-')
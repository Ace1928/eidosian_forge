import sys
import types
from testtools import TestCase
from testtools.matchers import (
from extras import (
def test_existing_module(self):
    result = try_imports(['os'], object())
    import os
    self.assertThat(result, Is(os))
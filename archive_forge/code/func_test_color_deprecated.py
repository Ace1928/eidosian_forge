import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def test_color_deprecated(self):
    with self.assertWarnsRegex(DeprecationWarning, "'Color' in 'traits'"):
        Color()
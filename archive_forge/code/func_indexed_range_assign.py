import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def indexed_range_assign(self, list, index1, index2, value):
    list[index1:index2] = value
import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def extended_slice_assign(self, list, index1, index2, step, value):
    list[index1:index2:step] = value
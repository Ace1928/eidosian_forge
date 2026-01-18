import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def del_extended_slice(self, list, index1, index2, step):
    del list[index1:index2:step]
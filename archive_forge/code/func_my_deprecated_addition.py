import unittest
from traits.testing.api import UnittestTools
from traits.util.api import deprecated
@deprecated('Addition is deprecated; use subtraction instead.')
def my_deprecated_addition(x, y):
    return x + y
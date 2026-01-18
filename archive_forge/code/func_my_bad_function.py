import unittest
from traits.testing.api import UnittestTools
from traits.util.api import deprecated
@deprecated('Broken code. Use something else.')
def my_bad_function():
    1 / 0
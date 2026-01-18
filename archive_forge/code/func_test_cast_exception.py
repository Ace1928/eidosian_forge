import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import (
from aniso8601.tests.compat import mock
def test_cast_exception(self):
    with self.assertRaises(ISOFormatError):
        cast('asdf', int)
    with self.assertRaises(ISOFormatError):
        cast('asdf', float)
from unittest import TestCase
import textwrap
from jsonschema import exceptions
from jsonschema.validators import _LATEST_VERSION
def test_multiple_errors_with_instance(self):
    e1, e2 = (exceptions.ValidationError('1', validator='foo', path=['bar', 'bar2'], instance='i1'), exceptions.ValidationError('2', validator='quux', path=['foobar', 2], instance='i2'))
    exceptions.ErrorTree([e1, e2])
from testtools import (
from testtools.compat import (
from testtools.matchers import (
from testtools.matchers._impl import (
from testtools.tests.helpers import FullStackRunTest
def test_forwards_details(self):
    x = Mismatch('description', {'foo': 'bar'})
    decorated = MismatchDecorator(x)
    self.assertEqual(x.get_details(), decorated.get_details())
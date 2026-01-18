from testtools import (
from testtools.compat import (
from testtools.matchers import (
from testtools.matchers._impl import (
from testtools.tests.helpers import FullStackRunTest
def test_forwards_description(self):
    x = Mismatch('description', {'foo': 'bar'})
    decorated = MismatchDecorator(x)
    self.assertEqual(x.describe(), decorated.describe())
from tempfile import NamedTemporaryFile
from testtools import TestCase
from subunit import read_test_list
from subunit.filters import find_stream
def test_no_argv(self):
    self.assertEqual('foo', find_stream('foo', []))
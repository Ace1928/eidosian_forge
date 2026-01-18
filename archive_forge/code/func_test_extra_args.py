import pickle
import re
from typing import List, Tuple
from .. import lazy_regex, tests
def test_extra_args(self):
    """Test that extra arguments are also properly passed"""
    pattern = lazy_regex.lazy_compile('foo', re.I)
    self.assertIsInstance(pattern, lazy_regex.LazyRegex)
    self.assertTrue(pattern.match('foo'))
    self.assertTrue(pattern.match('Foo'))
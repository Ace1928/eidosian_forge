import pickle
import re
from typing import List, Tuple
from .. import lazy_regex, tests
def test_findall(self):
    pattern = lazy_regex.lazy_compile('fo*')
    self.assertEqual(['f', 'fo', 'foo', 'fooo'], pattern.findall('f fo foo fooo'))
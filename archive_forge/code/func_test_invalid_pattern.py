import pickle
import re
from typing import List, Tuple
from .. import lazy_regex, tests
def test_invalid_pattern(self):
    error = lazy_regex.InvalidPattern('Bad pattern msg.')
    self.assertEqualDiff('Invalid pattern(s) found. Bad pattern msg.', str(error))
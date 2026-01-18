import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_attribute_contains(self):
    self.assert_select_multiple(('[rel*="style"]', ['l1']), ('link[rel*="style"]', ['l1']), ('notlink[rel*="notstyle"]', []), ('[rel*="notstyle"]', []), ('link[rel*="notstyle"]', []), ('link[href*="bla"]', ['l1']), ('[href*="http://"]', ['bob', 'me']), ('[id*="p"]', ['pmulti', 'p1']), ('div[id*="m"]', ['main']), ('a[id*="m"]', ['me']), ('[href*=".css"]', ['l1']), ('link[href*=".css"]', ['l1']), ('link[id*="1"]', ['l1']), ('[id*="1"]', ['data1', 'l1', 'p1', 'header1', 's1a1', 's1a2', 's2a1', 's1a2s1', 'dash1']), ('div[id*="1"]', ['data1']), ('[id*="noending"]', []), ('[href*="."]', ['bob', 'me', 'l1']), ('a[href*="."]', ['bob', 'me']), ('link[href*="."]', ['l1']), ('div[id*="n"]', ['main', 'inner']), ('div[id*="nn"]', ['inner']), ('div[data-tag*="edval"]', ['data1']))
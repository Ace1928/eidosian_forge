import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_general_sibling_selector(self):
    self.assert_selects('#p1 ~ h2', ['header2', 'header3'])
    self.assert_selects('#p1 ~ #header2', ['header2'])
    self.assert_selects('#p1 ~ h2 + a', ['me'])
    self.assert_selects('#p1 ~ h2 + [rel="me"]', ['me'])
    assert [] == self.soup.select('#inner ~ h2')
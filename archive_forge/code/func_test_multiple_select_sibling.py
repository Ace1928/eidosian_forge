import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_multiple_select_sibling(self):
    self.assert_selects('x, y ~ p[lang=fr]', ['xid', 'lang-fr'])
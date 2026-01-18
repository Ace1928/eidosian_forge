import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_multiple_select_nested(self):
    self.assert_selects('body > div > x, y > z', ['xid', 'zidb'])
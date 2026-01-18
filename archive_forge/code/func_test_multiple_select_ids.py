import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_multiple_select_ids(self):
    self.assert_selects('x, y > z[id=zida], z[id=zidab], z[id=zidb]', ['xid', 'zidb', 'zidab'])
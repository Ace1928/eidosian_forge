import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_multiple_select_indirect_descendant(self):
    self.assert_selects('div x,y,  z', ['xid', 'yid', 'zida', 'zidb', 'zidab', 'zidac'])
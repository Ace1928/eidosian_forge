import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_multiple_select_with_no_space(self):
    self.assert_selects('x,y', ['xid', 'yid'])
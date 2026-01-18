import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_invalid_multiple_select(self):
    with pytest.raises(SelectorSyntaxError):
        self.soup.select(',x, y')
    with pytest.raises(SelectorSyntaxError):
        self.soup.select('x,,y')
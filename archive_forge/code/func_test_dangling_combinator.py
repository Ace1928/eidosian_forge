import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_dangling_combinator(self):
    with pytest.raises(SelectorSyntaxError):
        self.soup.select('h1 >')
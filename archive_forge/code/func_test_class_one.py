import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_class_one(self):
    for selector in ('.onep', 'p.onep', 'html p.onep'):
        els = self.soup.select(selector)
        assert len(els) == 1
        assert els[0].name == 'p'
        assert els[0]['class'] == ['onep']
import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_items_in_id(self):
    els = self.soup.select('div#inner p')
    assert len(els) == 3
    for el in els:
        assert el.name == 'p'
    assert els[1]['class'] == ['onep']
    assert not els[0].has_attr('class')
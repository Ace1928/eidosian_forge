import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_class_mismatched_tag(self):
    els = self.soup.select('div.onep')
    assert len(els) == 0
import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_a_bunch_of_emptys(self):
    for selector in ('div#main del', 'div#main div.oops', 'div div#main'):
        assert len(self.soup.select(selector)) == 0
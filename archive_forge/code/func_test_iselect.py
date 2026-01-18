import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_iselect(self):
    gen = self.soup.css.iselect('h2')
    assert isinstance(gen, types.GeneratorType)
    [header2, header3] = gen
    assert header2['id'] == 'header2'
    assert header3['id'] == 'header3'
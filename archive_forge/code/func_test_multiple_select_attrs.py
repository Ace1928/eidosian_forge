import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_multiple_select_attrs(self):
    self.assert_selects('p[lang=en], p[lang=en-gb]', ['lang-en', 'lang-en-gb'])
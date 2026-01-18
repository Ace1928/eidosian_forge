import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_multi_class_selection(self):
    for selector in ('.class1.class3', '.class3.class2', '.class1.class2.class3'):
        self.assert_selects(selector, ['pmulti'])
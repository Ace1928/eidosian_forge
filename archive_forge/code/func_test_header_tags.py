import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_header_tags(self):
    self.assert_select_multiple(('h1', ['header1']), ('h2', ['header2', 'header3']))
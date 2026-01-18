import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_attribute_exists(self):
    self.assert_select_multiple(('[rel]', ['l1', 'bob', 'me']), ('link[rel]', ['l1']), ('a[rel]', ['bob', 'me']), ('[lang]', ['lang-en', 'lang-en-gb', 'lang-en-us', 'lang-fr']), ('p[class]', ['p1', 'pmulti']), ('[blah]', []), ('p[blah]', []), ('div[data-tag]', ['data1']))
import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def assert_selects(self, selector, expected_ids, **kwargs):
    results = self.soup.select(selector, **kwargs)
    assert isinstance(results, ResultSet)
    el_ids = [el['id'] for el in results]
    el_ids.sort()
    expected_ids.sort()
    assert expected_ids == el_ids, 'Selector %s, expected [%s], got [%s]' % (selector, ', '.join(expected_ids), ', '.join(el_ids))
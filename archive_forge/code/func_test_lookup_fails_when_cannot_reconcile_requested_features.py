import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from . import (
def test_lookup_fails_when_cannot_reconcile_requested_features(self):
    builder1 = self.builder_for_features('foo', 'bar')
    builder2 = self.builder_for_features('foo', 'baz')
    assert self.registry.lookup('bar', 'baz') is None
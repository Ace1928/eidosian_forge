import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from . import (
def test_lookup_fails_when_no_tree_builders_registered(self):
    assert self.registry.lookup() is None
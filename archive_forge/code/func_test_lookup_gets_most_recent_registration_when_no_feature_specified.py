import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from . import (
def test_lookup_gets_most_recent_registration_when_no_feature_specified(self):
    builder1 = self.builder_for_features('foo')
    builder2 = self.builder_for_features('bar')
    assert self.registry.lookup() == builder2
import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def soup(self, markup, **kwargs):
    """Build a Beautiful Soup object from markup."""
    builder = kwargs.pop('builder', self.default_builder)
    return BeautifulSoup(markup, builder=builder, **kwargs)
import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def assert_selects_ids(self, tags, should_match):
    """Make sure that the given tags have the correct IDs.

        This is used in tests that define a bunch of tags, each
        containing a single string, and then select certain strings by
        some mechanism.
        """
    assert [tag['id'] for tag in tags] == should_match
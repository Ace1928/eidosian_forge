import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def assertConnectedness(self, element):
    """Ensure that next_element and previous_element are properly
        set for all descendants of the given element.
        """
    earlier = None
    for e in element.descendants:
        if earlier:
            assert e == earlier.next_element
            assert earlier == e.previous_element
        earlier = e
import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_copy_preserves_encoding(self):
    soup = BeautifulSoup(b'<p>&nbsp;</p>', 'html.parser')
    encoding = soup.original_encoding
    copy = soup.__copy__()
    assert '<p>\xa0</p>' == str(copy)
    assert encoding == copy.original_encoding
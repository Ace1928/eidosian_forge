import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_copy_tag_preserves_namespace(self):
    xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<w:document xmlns:w="http://example.com/ns0"/>'
    soup = self.soup(xml)
    tag = soup.document
    duplicate = copy.copy(tag)
    assert tag.prefix == duplicate.prefix
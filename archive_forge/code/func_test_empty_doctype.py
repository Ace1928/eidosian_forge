import pickle
import pytest
import re
import warnings
from . import LXML_PRESENT, LXML_VERSION
from bs4 import (
from bs4.element import Comment, Doctype, SoupStrainer
from . import (
@pytest.mark.skipif(not LXML_PRESENT or LXML_VERSION < (2, 3, 5, 0), reason='Skipping doctype test for old version of lxml to avoid segfault.')
def test_empty_doctype(self):
    soup = self.soup('<!DOCTYPE>')
    doctype = soup.contents[0]
    assert '' == doctype.strip()
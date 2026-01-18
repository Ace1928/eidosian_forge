import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_formatter_skips_style_tag_for_html_documents(self):
    doc = '\n  <style type="text/css">\n   console.log("< < hey > > ");\n  </style>\n'
    encoded = BeautifulSoup(doc, 'html.parser').encode()
    assert b'< < hey > >' in encoded
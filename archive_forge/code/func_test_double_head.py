import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_double_head(self):
    html = '<!DOCTYPE html>\n<html>\n<head>\n<title>Ordinary HEAD element test</title>\n</head>\n<script type="text/javascript">\nalert("Help!");\n</script>\n<body>\nHello, world!\n</body>\n</html>\n'
    soup = self.soup(html)
    assert 'text/javascript' == soup.find('script')['type']
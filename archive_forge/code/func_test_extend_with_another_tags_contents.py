from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
@pytest.mark.parametrize('get_tags', [lambda tag: tag, lambda tag: tag.contents])
def test_extend_with_another_tags_contents(self, get_tags):
    data = '<body><div id="d1"><a>1</a><a>2</a><a>3</a><a>4</a></div><div id="d2"></div></body>'
    soup = self.soup(data)
    d1 = soup.find('div', id='d1')
    d2 = soup.find('div', id='d2')
    tags = get_tags(d1)
    d2.extend(tags)
    assert '<div id="d1"></div>' == d1.decode()
    assert '<div id="d2"><a>1</a><a>2</a><a>3</a><a>4</a></div>' == d2.decode()
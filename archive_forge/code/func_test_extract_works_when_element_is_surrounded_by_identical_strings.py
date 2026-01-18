from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_extract_works_when_element_is_surrounded_by_identical_strings(self):
    soup = self.soup('<html>\n<body>hi</body>\n</html>')
    soup.find('body').extract()
    assert None == soup.find('body')
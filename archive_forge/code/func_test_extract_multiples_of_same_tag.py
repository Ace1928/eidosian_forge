from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_extract_multiples_of_same_tag(self):
    soup = self.soup('\n<html>\n<head>\n<script>foo</script>\n</head>\n<body>\n <script>bar</script>\n <a></a>\n</body>\n<script>baz</script>\n</html>')
    [soup.script.extract() for i in soup.find_all('script')]
    assert '<body>\n\n<a></a>\n</body>' == str(soup.body)
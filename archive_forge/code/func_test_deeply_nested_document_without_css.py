import os
import pytest
from bs4 import (
@pytest.mark.parametrize('filename', ['clusterfuzz-testcase-minimized-bs4_fuzzer-5984173902397440', 'clusterfuzz-testcase-minimized-bs4_fuzzer-5167584867909632', 'clusterfuzz-testcase-minimized-bs4_fuzzer-6124268085182464', 'clusterfuzz-testcase-minimized-bs4_fuzzer-6450958476902400'])
def test_deeply_nested_document_without_css(self, filename):
    markup = self.__markup(filename)
    BeautifulSoup(markup, 'html.parser').encode()
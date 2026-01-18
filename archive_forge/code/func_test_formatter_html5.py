import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_formatter_html5(self):
    markup = '<br><b>&lt;&lt;Sacré bleu!&gt;&gt;</b>'
    soup = self.soup(markup)
    decoded = soup.decode(formatter='html5')
    assert decoded == self.document_for('<br><b>&lt;&lt;Sacr&eacute; bleu!&gt;&gt;</b>')
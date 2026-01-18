import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_multiple_copies_of_a_tag(self):
    """Prevent recurrence of a bug in the html5lib treebuilder."""
    content = '<!DOCTYPE html>\n<html>\n <body>\n   <article id="a" >\n   <div><a href="1"></div>\n   <footer>\n     <a href="2"></a>\n   </footer>\n  </article>\n  </body>\n</html>\n'
    soup = self.soup(content)
    self.assertConnectedness(soup.article)
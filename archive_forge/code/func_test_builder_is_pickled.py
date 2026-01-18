from pdb import set_trace
import pickle
import pytest
import warnings
from bs4.builder import (
from bs4.builder._htmlparser import BeautifulSoupHTMLParser
from . import SoupTest, HTMLTreeBuilderSmokeTest
def test_builder_is_pickled(self):
    """Unlike most tree builders, HTMLParserTreeBuilder and will
        be restored after pickling.
        """
    tree = self.soup('<a><b>foo</a>')
    dumped = pickle.dumps(tree, 2)
    loaded = pickle.loads(dumped)
    assert isinstance(loaded.builder, type(tree.builder))
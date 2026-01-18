import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from . import (
def test_named_library(self):
    if LXML_PRESENT:
        assert registry.lookup('lxml', 'xml') == LXMLTreeBuilderForXML
        assert registry.lookup('lxml', 'html') == LXMLTreeBuilder
    if HTML5LIB_PRESENT:
        assert registry.lookup('html5lib') == HTML5TreeBuilder
    assert registry.lookup('html.parser') == HTMLParserTreeBuilder
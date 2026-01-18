import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from . import (
def test_lookup_by_markup_type(self):
    if LXML_PRESENT:
        assert registry.lookup('html') == LXMLTreeBuilder
        assert registry.lookup('xml') == LXMLTreeBuilderForXML
    else:
        assert registry.lookup('xml') == None
        if HTML5LIB_PRESENT:
            assert registry.lookup('html') == HTML5TreeBuilder
        else:
            assert registry.lookup('html') == HTMLParserTreeBuilder
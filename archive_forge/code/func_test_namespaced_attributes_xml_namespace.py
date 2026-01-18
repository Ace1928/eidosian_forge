import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_namespaced_attributes_xml_namespace(self):
    markup = '<foo xml:lang="fr">bar</foo>'
    soup = self.soup(markup)
    assert str(soup.foo) == markup
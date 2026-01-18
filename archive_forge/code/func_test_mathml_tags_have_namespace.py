import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_mathml_tags_have_namespace(self):
    markup = '<math><msqrt>5</msqrt></math>'
    soup = self.soup(markup)
    namespace = 'http://www.w3.org/1998/Math/MathML'
    assert namespace == soup.math.namespace
    assert namespace == soup.msqrt.namespace
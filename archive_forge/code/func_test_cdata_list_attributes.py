from pdb import set_trace
import logging
import os
import pickle
import pytest
import sys
import tempfile
from bs4 import (
from bs4.builder import (
from bs4.element import (
from . import (
import warnings
def test_cdata_list_attributes(self):
    markup = '<a id=" an id " class=" a class "></a>'
    soup = self.soup(markup)
    a = soup.a
    assert ' an id ' == a['id']
    assert ['a', 'class'] == a['class']
    soup = self.soup(markup, builder=default_builder, multi_valued_attributes=None)
    assert ' a class ' == soup.a['class']
    for switcheroo in ({'*': 'id'}, {'a': 'id'}):
        with warnings.catch_warnings(record=True) as w:
            soup = self.soup(markup, builder=None, multi_valued_attributes=switcheroo)
        a = soup.a
        assert ['an', 'id'] == a['id']
        assert ' a class ' == a['class']
import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_invalid_doctype(self):
    markup = '<![if word]>content<![endif]>'
    markup = '<!DOCTYPE html]ff>'
    soup = self.soup(markup)
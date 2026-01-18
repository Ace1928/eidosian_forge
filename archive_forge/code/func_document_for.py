import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def document_for(self, markup, **kwargs):
    """Turn an HTML fragment into a document.

        The details depend on the builder.
        """
    return self.default_builder(**kwargs).test_fragment_to_document(markup)
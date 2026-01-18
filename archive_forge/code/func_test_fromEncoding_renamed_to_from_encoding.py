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
def test_fromEncoding_renamed_to_from_encoding(self):
    with warnings.catch_warnings(record=True) as w:
        utf8 = b'\xc3\xa9'
        soup = BeautifulSoup(utf8, 'html.parser', fromEncoding='utf8')
    warning = self._assert_warning(w, DeprecationWarning)
    msg = str(warning.message)
    assert 'fromEncoding' in msg
    assert 'from_encoding' in msg
    assert 'utf8' == soup.original_encoding
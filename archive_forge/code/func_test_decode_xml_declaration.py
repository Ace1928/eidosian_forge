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
@pytest.mark.parametrize('eventual_encoding,actual_encoding', [('utf-8', 'utf-8'), ('utf-16', 'utf-16')])
def test_decode_xml_declaration(self, eventual_encoding, actual_encoding):
    soup = self.soup('<tag></tag>')
    soup.is_xml = True
    assert f'<?xml version="1.0" encoding="{actual_encoding}"?>\n<tag></tag>' == soup.decode(eventual_encoding=eventual_encoding)
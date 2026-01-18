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
def test_no_warning_if_explicit_parser_specified(self):
    with warnings.catch_warnings(record=True) as w:
        soup = self.soup('<a><b></b></a>')
    assert [] == w
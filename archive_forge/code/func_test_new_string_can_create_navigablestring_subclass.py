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
def test_new_string_can_create_navigablestring_subclass(self):
    soup = self.soup('')
    s = soup.new_string('foo', Comment)
    assert 'foo' == s
    assert isinstance(s, Comment)
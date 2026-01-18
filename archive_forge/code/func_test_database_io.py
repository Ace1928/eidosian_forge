from __future__ import absolute_import, unicode_literals
import pickle
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from io import BytesIO, TextIOWrapper
import six
import pytest
from pybtex.database import parse_bytes, parse_string, BibliographyData, Entry
from pybtex.plugin import find_plugin
from .data import reference_data
@pytest.mark.parametrize(['io_cls'], [(PybtexBytesIO,), (PybtexStringIO,), (PybtexEntryStringIO,), (PybtexBytesIO,)])
@pytest.mark.parametrize(['bib_format'], [('bibtex',), ('bibtexml',), ('yaml',)])
def test_database_io(io_cls, bib_format):
    check_database_io(io_cls(bib_format))
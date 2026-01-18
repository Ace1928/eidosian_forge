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
def check_database_io(io_obj):
    serialized_data = io_obj.serialize(io_obj.reference_data)
    deserialized_data = io_obj.deserialize(serialized_data)
    assert deserialized_data == io_obj.reference_data
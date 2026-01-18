import os
import sys
from inspect import cleandoc
from itertools import chain
from string import ascii_letters, digits
from unittest import mock
import numpy as np
import pytest
import shapely
from shapely.decorators import multithreading_enabled, requires_geos
@pytest.fixture
def sphinx_doc_build():
    os.environ['SPHINX_DOC_BUILD'] = '1'
    yield
    del os.environ['SPHINX_DOC_BUILD']
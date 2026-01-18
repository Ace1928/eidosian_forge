import os
import shutil
from tempfile import mkdtemp
import pytest
import numpy
import py.path as pp
@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace['np'] = numpy
    doctest_namespace['os'] = os
    doctest_namespace['pytest'] = pytest
    doctest_namespace['datadir'] = data_dir
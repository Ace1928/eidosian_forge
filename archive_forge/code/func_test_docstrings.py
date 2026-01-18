import os
import unittest
import doctest
from pygsp.tests import test_graphs, test_filters
from pygsp.tests import test_utils, test_plotting
def test_docstrings(root, ext, setup=None):
    files = list(gen_recursive_file(root, ext))
    return doctest.DocFileSuite(*files, setUp=setup, module_relative=False)
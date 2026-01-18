import os
import sys
import warnings
from importlib.metadata import entry_points
import pytest
import networkx
@pytest.fixture(autouse=True)
def add_nx(doctest_namespace):
    doctest_namespace['nx'] = networkx
    try:
        import numpy as np
        np.set_printoptions(legacy='1.21')
    except ImportError:
        pass
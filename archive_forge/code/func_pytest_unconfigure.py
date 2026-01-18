import pytest
import sys
import matplotlib
from matplotlib import _api
def pytest_unconfigure(config):
    matplotlib._called_from_pytest = False
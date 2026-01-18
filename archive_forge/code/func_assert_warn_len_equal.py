import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def assert_warn_len_equal(mod, n_in_context):
    try:
        mod_warns = mod.__warningregistry__
    except AttributeError:
        mod_warns = {}
    num_warns = len(mod_warns)
    if 'version' in mod_warns:
        num_warns -= 1
    assert_equal(num_warns, n_in_context)
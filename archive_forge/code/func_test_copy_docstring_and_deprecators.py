import difflib
import numpy as np
import sys
from pathlib import Path
import pytest
import matplotlib as mpl
from matplotlib.testing import subprocess_run_for_testing
from matplotlib import pyplot as plt
def test_copy_docstring_and_deprecators(recwarn):

    @mpl._api.rename_parameter('(version)', 'old', 'new')
    @mpl._api.make_keyword_only('(version)', 'kwo')
    def func(new, kwo=None):
        pass

    @plt._copy_docstring_and_deprecators(func)
    def wrapper_func(new, kwo=None):
        pass
    wrapper_func(None)
    wrapper_func(new=None)
    wrapper_func(None, kwo=None)
    wrapper_func(new=None, kwo=None)
    assert not recwarn
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        wrapper_func(old=None)
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        wrapper_func(None, None)
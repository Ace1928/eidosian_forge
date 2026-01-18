import sys
import warnings
from functools import partial
from textwrap import indent
import pytest
from nibabel.deprecator import (
from ..testing import clear_and_catch_warnings
def func_doc(i):
    """A docstring"""
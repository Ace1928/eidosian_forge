import sys
import warnings
from functools import partial
from textwrap import indent
import pytest
from nibabel.deprecator import (
from ..testing import clear_and_catch_warnings
def test__ensure_cr():
    assert _ensure_cr('  foo') == '  foo\n'
    assert _ensure_cr('  foo\n') == '  foo\n'
    assert _ensure_cr('  foo  ') == '  foo\n'
    assert _ensure_cr('foo  ') == 'foo\n'
    assert _ensure_cr('foo  \n bar') == 'foo  \n bar\n'
    assert _ensure_cr('foo  \n\n') == 'foo\n'
import sys
import warnings
from functools import partial
from textwrap import indent
import pytest
from nibabel.deprecator import (
from ..testing import clear_and_catch_warnings
def test__add_dep_doc():
    assert _add_dep_doc('', 'foo') == 'foo\n'
    assert _add_dep_doc('bar', 'foo') == 'bar\n\nfoo\n'
    assert _add_dep_doc('   bar', 'foo') == '   bar\n\nfoo\n'
    assert _add_dep_doc('   bar', 'foo\n') == '   bar\n\nfoo\n'
    assert _add_dep_doc('bar\n\n', 'foo') == 'bar\n\nfoo\n'
    assert _add_dep_doc('bar\n    \n', 'foo') == 'bar\n\nfoo\n'
    assert _add_dep_doc(' bar\n\nSome explanation', 'foo\nbaz') == ' bar\n\nfoo\nbaz\n\nSome explanation\n'
    assert _add_dep_doc(' bar\n\n  Some explanation', 'foo\nbaz') == ' bar\n  \n  foo\n  baz\n  \n  Some explanation\n'
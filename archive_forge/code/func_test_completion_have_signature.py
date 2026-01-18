import os
import pytest
import sys
import textwrap
import unittest
from contextlib import contextmanager
from traitlets.config.loader import Config
from IPython import get_ipython
from IPython.core import completer
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.generics import complete_object
from IPython.testing import decorators as dec
from IPython.core.completer import (
@pytest.mark.xfail(sys.version_info.releaselevel in ('alpha',), reason='Parso does not yet parse 3.13')
def test_completion_have_signature(self):
    """
        Lets make sure jedi is capable of pulling out the signature of the function we are completing.
        """
    ip = get_ipython()
    with provisionalcompleter():
        ip.Completer.use_jedi = True
        completions = ip.Completer.completions('ope', 3)
        c = next(completions)
        ip.Completer.use_jedi = False
    assert 'file' in c.signature, 'Signature of function was not found by completer'
    assert 'encoding' in c.signature, 'Signature of function was not found by completer'
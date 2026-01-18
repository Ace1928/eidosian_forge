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
def test_jedi(self):
    """
        A couple of issue we had with Jedi
        """
    ip = get_ipython()

    def _test_complete(reason, s, comp, start=None, end=None):
        l = len(s)
        start = start if start is not None else l
        end = end if end is not None else l
        with provisionalcompleter():
            ip.Completer.use_jedi = True
            completions = set(ip.Completer.completions(s, l))
            ip.Completer.use_jedi = False
            assert Completion(start, end, comp) in completions, reason

    def _test_not_complete(reason, s, comp):
        l = len(s)
        with provisionalcompleter():
            ip.Completer.use_jedi = True
            completions = set(ip.Completer.completions(s, l))
            ip.Completer.use_jedi = False
            assert Completion(l, l, comp) not in completions, reason
    import jedi
    jedi_version = tuple((int(i) for i in jedi.__version__.split('.')[:3]))
    if jedi_version > (0, 10):
        _test_complete('jedi >0.9 should complete and not crash', 'a=1;a.', 'real')
    _test_complete('can infer first argument', 'a=(1,"foo");a[0].', 'real')
    _test_complete('can infer second argument', 'a=(1,"foo");a[1].', 'capitalize')
    _test_complete('cover duplicate completions', 'im', 'import', 0, 2)
    _test_not_complete('does not mix types', 'a=(1,"foo");a[0].', 'capitalize')
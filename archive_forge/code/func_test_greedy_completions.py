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
def test_greedy_completions(self):
    """
        Test the capability of the Greedy completer.

        Most of the test here does not really show off the greedy completer, for proof
        each of the text below now pass with Jedi. The greedy completer is capable of more.

        See the :any:`test_dict_key_completion_contexts`

        """
    ip = get_ipython()
    ip.ex('a=list(range(5))')
    ip.ex("d = {'a b': str}")
    _, c = ip.complete('.', line='a[0].')
    self.assertFalse('.real' in c, "Shouldn't have completed on a[0]: %s" % c)

    def _(line, cursor_pos, expect, message, completion):
        with greedy_completion(), provisionalcompleter():
            ip.Completer.use_jedi = False
            _, c = ip.complete('.', line=line, cursor_pos=cursor_pos)
            self.assertIn(expect, c, message % c)
            ip.Completer.use_jedi = True
            with provisionalcompleter():
                completions = ip.Completer.completions(line, cursor_pos)
            self.assertIn(completion, completions)
    with provisionalcompleter():
        _('a[0].', 5, '.real', 'Should have completed on a[0].: %s', Completion(5, 5, 'real'))
        _('a[0].r', 6, '.real', 'Should have completed on a[0].r: %s', Completion(5, 6, 'real'))
        _('a[0].from_', 10, '.from_bytes', 'Should have completed on a[0].from_: %s', Completion(5, 10, 'from_bytes'))
        _('assert str.star', 14, 'str.startswith', 'Should have completed on `assert str.star`: %s', Completion(11, 14, 'startswith'))
        _("d['a b'].str", 12, '.strip', "Should have completed on `d['a b'].str`: %s", Completion(9, 12, 'strip'))
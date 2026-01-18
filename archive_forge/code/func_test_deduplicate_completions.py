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
@pytest.mark.xfail(reason='Known failure on jedi<=0.18.0')
def test_deduplicate_completions(self):
    """
        Test that completions are correctly deduplicated (even if ranges are not the same)
        """
    ip = get_ipython()
    ip.ex(textwrap.dedent('\n        class Z:\n            zoo = 1\n        '))
    with provisionalcompleter():
        ip.Completer.use_jedi = True
        l = list(_deduplicate_completions('Z.z', ip.Completer.completions('Z.z', 3)))
        ip.Completer.use_jedi = False
    assert len(l) == 1, 'Completions (Z.z<tab>) correctly deduplicate: %s ' % l
    assert l[0].text == 'zoo'
import os
import shutil
import sys
import tempfile
import unittest
from os.path import join
from tempfile import TemporaryDirectory
from IPython.core.completerlib import magic_run_completer, module_completion, try_import
from IPython.testing.decorators import onlyif_unicode_paths
def test_bad_module_all():
    """Test module with invalid __all__

    https://github.com/ipython/ipython/issues/9678
    """
    testsdir = os.path.dirname(__file__)
    sys.path.insert(0, testsdir)
    try:
        results = module_completion('from bad_all import ')
        assert 'puppies' in results
        for r in results:
            assert isinstance(r, str)
        results = module_completion('import bad_all.')
        assert results == []
    finally:
        sys.path.remove(testsdir)
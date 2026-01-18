import errno
import os
import shutil
import sys
from .. import tests, ui
from ..clean_tree import clean_tree, iter_deletables
from ..controldir import ControlDir
from ..osutils import supports_symlinks
from . import TestCaseInTempDir
def test_delete_items_warnings(self):
    """Ensure delete_items issues warnings on EACCES. (bug #430785)
        """

    def _dummy_unlink(path):
        """unlink() files other than files named '0foo'.
            """
        if path.endswith('0foo'):
            e = OSError()
            e.errno = errno.EACCES
            raise e

    def _dummy_rmtree(path, ignore_errors=False, onerror=None):
        """Call user supplied error handler onerror.
            """
        try:
            raise OSError
        except OSError as e:
            e.errno = errno.EACCES
            excinfo = sys.exc_info()
            function = os.remove
            if 'subdir0' not in path:
                function = os.listdir
            onerror(function=function, path=path, excinfo=excinfo)
    self.overrideAttr(os, 'unlink', _dummy_unlink)
    self.overrideAttr(shutil, 'rmtree', _dummy_rmtree)
    ui.ui_factory = tests.TestUIFactory()
    stderr = ui.ui_factory.stderr
    ControlDir.create_standalone_workingtree('.')
    self.build_tree(['0foo', '1bar', '2baz', 'subdir0/'])
    clean_tree('.', unknown=True, no_prompt=True)
    self.assertContainsRe(stderr.getvalue(), 'bzr: warning: unable to remove.*0foo')
    self.assertContainsRe(stderr.getvalue(), 'bzr: warning: unable to remove.*subdir0')
    self.build_tree(['subdir1/'])
    self.assertRaises(OSError, clean_tree, '.', unknown=True, no_prompt=True)
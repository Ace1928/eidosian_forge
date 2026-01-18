import os
import re
import unittest
from breezy import bzr, config, controldir, errors, osutils, repository, tests
from breezy.bzr.groupcompress_repo import RepositoryFormat2a
def test_exception_exitcode(self):
    out, err = self.run_brz_subprocess(['assert-fail'], universal_newlines=True, retcode=errors.EXIT_INTERNAL_ERROR)
    self.assertEqual(4, errors.EXIT_INTERNAL_ERROR)
    self.assertContainsRe(err, b'\\nAssertionError: always fails\\n')
    self.assertContainsRe(err, b'Breezy has encountered an internal error')
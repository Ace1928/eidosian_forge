import os
import re
import unittest
from breezy import bzr, config, controldir, errors, osutils, repository, tests
from breezy.bzr.groupcompress_repo import RepositoryFormat2a
def check_warning(self, present):
    if present:
        check = self.assertContainsRe
    else:
        check = self.assertNotContainsRe
    check(self.get_log(), 'WARNING.*brz upgrade')
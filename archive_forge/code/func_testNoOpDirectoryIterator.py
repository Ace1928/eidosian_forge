from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
import six
import tempfile
from gslib import wildcard_iterator
from gslib.exception import InvalidUrlError
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetDummyProjectForUnitTest
def testNoOpDirectoryIterator(self):
    """Tests that directory-only URI iterates just that one URI."""
    results = list(self._test_wildcard_iterator(suri(tempfile.tempdir)).IterAll(expand_top_level_buckets=True))
    self.assertEqual(1, len(results))
    self.assertEqual(suri(tempfile.tempdir), str(results[0]))
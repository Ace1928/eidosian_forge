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
def testMatchingFileSubset(self):
    """Tests matching a subset of files, based on wildcard."""
    exp_uri_strs = set([suri(self.test_dir, 'abcd'), suri(self.test_dir, 'abdd')])
    uri = self._test_storage_uri(suri(self.test_dir, 'ab??'))
    actual_uri_strs = set((str(u) for u in self._test_wildcard_iterator(uri).IterAll(expand_top_level_buckets=True)))
    self.assertEqual(exp_uri_strs, actual_uri_strs)
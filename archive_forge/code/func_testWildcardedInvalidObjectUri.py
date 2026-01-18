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
def testWildcardedInvalidObjectUri(self):
    """Tests that we raise an exception for wildcarded invalid URI."""
    try:
        for unused_ in self._test_wildcard_iterator('badscheme://asdf').IterAll(expand_top_level_buckets=True):
            self.assertFalse('Expected InvalidUrlError not raised.')
    except InvalidUrlError as e:
        self.assertTrue(e.message.find('Unrecognized scheme') != -1)
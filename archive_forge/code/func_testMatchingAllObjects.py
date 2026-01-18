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
def testMatchingAllObjects(self):
    """Tests matching all objects, based on wildcard."""
    actual_obj_uri_strs = set((six.ensure_text(str(u)) for u in self._test_wildcard_iterator(self.test_bucket0_uri.clone_replace_name('**')).IterAll(expand_top_level_buckets=True)))
    self.assertEqual(self.test_bucket0_obj_uri_strs, actual_obj_uri_strs)
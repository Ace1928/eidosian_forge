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
def testMatchingAllFilesWithSize(self):
    """Tests matching all files, based on wildcard."""
    uri = self._test_storage_uri(suri(self.test_dir, '*'))
    blrs = self._test_wildcard_iterator(uri).IterAll(expand_top_level_buckets=True, bucket_listing_fields=['size'])
    num_expected_objects = 3
    num_actual_objects = 0
    for blr in blrs:
        self.assertTrue(str(blr) in self.immed_child_uri_strs)
        if blr.IsObject():
            num_actual_objects += 1
            self.assertEqual(blr.root_object.size, 6)
    self.assertEqual(num_expected_objects, num_actual_objects)
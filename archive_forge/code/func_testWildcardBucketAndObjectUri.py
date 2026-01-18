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
def testWildcardBucketAndObjectUri(self):
    """Tests matching with both bucket and object wildcards."""
    exp_obj_uri_strs = set([str(self.test_bucket0_uri.clone_replace_name('abcd'))])
    with SetDummyProjectForUnitTest():
        actual_obj_uri_strs = set((str(u) for u in self._test_wildcard_iterator('%s0*/abc*' % self.base_uri_str).IterAll(expand_top_level_buckets=True)))
    self.assertEqual(exp_obj_uri_strs, actual_obj_uri_strs)
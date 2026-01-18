from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import sys
from unittest import mock
from gslib.exception import NO_URLS_MATCHED_PREFIX
from gslib.exception import NO_URLS_MATCHED_TARGET
import gslib.tests.testcase as testcase
from gslib.tests.testcase.base import MAX_BUCKET_LENGTH
from gslib.tests.testcase.integration_testcase import SkipForS3
import gslib.tests.util as util
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils import shim_util
from gslib.utils.retry_util import Retry
def test_all_versions_current(self):
    """Test that 'rm -a' for an object with a current version works."""
    bucket_uri = self.CreateVersionedBucket()
    key_uri = self.StorageUriCloneReplaceName(bucket_uri, 'foo')
    self.StorageUriSetContentsFromString(key_uri, 'bar')
    g1 = urigen(key_uri)
    self.StorageUriSetContentsFromString(key_uri, 'baz')
    g2 = urigen(key_uri)

    def _Check1(stderr_lines):
        stderr = self.RunGsUtil(['-m', 'rm', '-a', suri(key_uri)], return_stderr=True)
        stderr_lines.update(set(stderr.splitlines()))
        stderr = '\n'.join(stderr_lines)
        self.assertEqual(stderr.count('Removing %s://' % self.default_provider), 2)
        self.assertIn('Removing %s#%s...' % (suri(key_uri), g1), stderr)
        self.assertIn('Removing %s#%s...' % (suri(key_uri), g2), stderr)
    all_stderr_lines = set()
    if self.multiregional_buckets:
        self.AssertNObjectsInBucket(bucket_uri, 2, versioned=True)

        @Retry(AssertionError, tries=3, timeout_secs=1)
        def _CheckWithRetries(stderr_lines):
            _Check1(stderr_lines)
        _CheckWithRetries(all_stderr_lines)
    else:
        _Check1(all_stderr_lines)
    self.AssertNObjectsInBucket(bucket_uri, 0, versioned=True)
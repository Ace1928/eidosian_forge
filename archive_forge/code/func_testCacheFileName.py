from __future__ import absolute_import
import datetime
import logging
import os
import stat
import sys
import unittest
from freezegun import freeze_time
from gcs_oauth2_boto_plugin import oauth2_client
import httplib2
def testCacheFileName(self):
    """Tests configuring the cache with a specific file name."""
    cache = oauth2_client.FileSystemTokenCache(path_pattern='/var/run/ccache/token.%(uid)s.%(key)s')
    if IS_WINDOWS:
        uid = '_'
    else:
        uid = os.getuid()
    self.assertEqual('/var/run/ccache/token.%s.abc123' % uid, cache.CacheFileName('abc123'))
    cache = oauth2_client.FileSystemTokenCache(path_pattern='/var/run/ccache/token.%(key)s')
    self.assertEqual('/var/run/ccache/token.abc123', cache.CacheFileName('abc123'))
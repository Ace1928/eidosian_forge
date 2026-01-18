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
def testGetBadFile(self):
    f = open(self.cache.CacheFileName(self.key), 'w')
    f.write('blah')
    f.close()
    self.assertEqual(None, self.cache.GetToken(self.key))
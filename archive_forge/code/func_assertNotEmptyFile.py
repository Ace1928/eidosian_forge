import http.client as http
import os
import stat
import httplib2
from glance.tests import functional
def assertNotEmptyFile(self, path):
    self.assertTrue(os.path.exists(path))
    self.assertNotEqual(os.stat(path)[stat.ST_SIZE], 0)
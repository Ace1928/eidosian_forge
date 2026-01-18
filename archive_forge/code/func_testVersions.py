import codecs
import gzip
import os
import six.moves.urllib.request as urllib_request
import tempfile
import unittest
from apitools.gen import util
from mock import patch
def testVersions(self):
    already_valid = 'v1'
    self.assertEqual(already_valid, util.NormalizeVersion(already_valid))
    to_clean = 'v0.1'
    self.assertEqual('v0_1', util.NormalizeVersion(to_clean))
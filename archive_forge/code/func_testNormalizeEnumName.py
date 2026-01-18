import codecs
import gzip
import os
import six.moves.urllib.request as urllib_request
import tempfile
import unittest
from apitools.gen import util
from mock import patch
def testNormalizeEnumName(self):
    names = util.Names([''])
    self.assertEqual('_0', names.NormalizeEnumName('0'))
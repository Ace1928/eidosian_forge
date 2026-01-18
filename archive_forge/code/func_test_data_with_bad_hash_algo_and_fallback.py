import errno
import hashlib
import testtools
from unittest import mock
import ddt
from glanceclient.common import utils as common_utils
from glanceclient import exc
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import images
def test_data_with_bad_hash_algo_and_fallback(self):
    body = self.controller.data('badalgo-db27-11e1-a1eb-080027cbe205', do_checksum=False, allow_md5_fallback=True)
    body = ''.join([b for b in body])
    self.assertEqual('BB', body)
    body = self.controller.data('badalgo-db27-11e1-a1eb-080027cbe205', allow_md5_fallback=True)
    body = ''.join([b for b in body])
    self.assertEqual('BB', body)
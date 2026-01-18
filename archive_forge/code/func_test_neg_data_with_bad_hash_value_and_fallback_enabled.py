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
def test_neg_data_with_bad_hash_value_and_fallback_enabled(self):
    body = self.controller.data('bad-multihash-value-good-checksum', allow_md5_fallback=False)
    try:
        body = ''.join([b for b in body])
        self.fail('bad os_hash_value did not raise an error.')
    except IOError as e:
        self.assertEqual(errno.EPIPE, e.errno)
        msg = 'expected badmultihashvalue'
        self.assertIn(msg, str(e))
    body = self.controller.data('bad-multihash-value-good-checksum', allow_md5_fallback=True)
    try:
        body = ''.join([b for b in body])
        self.fail('bad os_hash_value did not raise an error.')
    except IOError as e:
        self.assertEqual(errno.EPIPE, e.errno)
        msg = 'expected badmultihashvalue'
        self.assertIn(msg, str(e))
    body = self.controller.data('bad-multihash-value-good-checksum', do_checksum=False)
    body = ''.join([b for b in body])
    self.assertEqual('GOODCHECKSUM', body)
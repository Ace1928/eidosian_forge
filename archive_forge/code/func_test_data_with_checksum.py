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
def test_data_with_checksum(self):
    for prefix in ['headeronly', 'chkonly', 'multihash']:
        body = self.controller.data(prefix + '-dd57-11e1-af0f-02163e68b1d8', do_checksum=False)
        body = ''.join([b for b in body])
        self.assertEqual('CCC', body)
        body = self.controller.data(prefix + '-dd57-11e1-af0f-02163e68b1d8')
        body = ''.join([b for b in body])
        self.assertEqual('CCC', body)
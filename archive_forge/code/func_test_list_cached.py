import testtools
from unittest import mock
from glanceclient.common import utils as common_utils
from glanceclient import exc
from glanceclient.tests import utils
from glanceclient.v2 import cache
@mock.patch.object(common_utils, 'has_version')
def test_list_cached(self, mock_has_version):
    mock_has_version.return_value = True
    images = self.controller.list()
    self.assertEqual(2, len(images['cached_images']))
    self.assertEqual(2, len(images['queued_images']))
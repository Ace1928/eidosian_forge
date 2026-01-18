import testtools
from unittest import mock
from glanceclient.common import utils as common_utils
from glanceclient import exc
from glanceclient.tests import utils
from glanceclient.v2 import cache
@mock.patch.object(common_utils, 'has_version')
def test_cache_clear_with_header(self, mock_has_version):
    mock_has_version.return_value = True
    self.controller.clear('cache')
    expect = [('DELETE', '/v2/cache', {'x-image-cache-clear-target': 'cache'}, None)]
    self.assertEqual(expect, self.api.calls)
import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_find_resource_get_uuid(self):
    uuid = '9a0dc2a0-ad0d-11e3-a5e2-0800200c9a66'
    self.manager.get = mock.Mock(return_value=self.expected)
    result = utils.find_resource(self.manager, uuid)
    self.assertEqual(self.expected, result)
    self.manager.get.assert_called_with(uuid)
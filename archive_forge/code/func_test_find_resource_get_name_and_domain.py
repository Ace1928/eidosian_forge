import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_find_resource_get_name_and_domain(self):
    name = 'admin'
    domain_id = '30524568d64447fbb3fa8b7891c10dd6'
    side_effect = [Exception('Boom!'), self.expected]
    self.manager.get = mock.Mock(side_effect=side_effect)
    result = utils.find_resource(self.manager, name, domain_id=domain_id)
    self.assertEqual(self.expected, result)
    self.manager.get.assert_called_with(name, domain_id=domain_id)
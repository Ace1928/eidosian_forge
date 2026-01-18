import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_find_resource_silly_resource_no_unique_match(self):
    self.manager = mock.Mock()
    self.manager.get = mock.Mock(side_effect=Exception('Boom!'))
    self.manager.find = mock.Mock(side_effect=AttributeError("'Controller' object has no attribute 'find'"))
    silly_resource = FakeOddballResource(None, {'id': '12345', 'name': self.name}, loaded=True)
    silly_resource_same = FakeOddballResource(None, {'id': 'abcde', 'name': self.name}, loaded=True)
    self.manager.list = mock.Mock(return_value=[silly_resource, silly_resource_same])
    result = self.assertRaises(exceptions.CommandError, utils.find_resource, self.manager, self.name)
    self.assertEqual("More than one resource exists with the name or ID 'legos'.", str(result))
    self.manager.get.assert_called_with(self.name)
    self.manager.find.assert_called_with(name=self.name)
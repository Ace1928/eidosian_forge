import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_find_resource_find_not_found(self):
    self.manager.get = mock.Mock(side_effect=Exception('Boom!'))
    self.manager.find = mock.Mock(side_effect=exceptions.NotFound(404, '2'))
    result = self.assertRaises(exceptions.CommandError, utils.find_resource, self.manager, self.name)
    self.assertEqual("No lego with a name or ID of 'legos' exists.", str(result))
    self.manager.get.assert_called_with(self.name)
    self.manager.find.assert_called_with(name=self.name)
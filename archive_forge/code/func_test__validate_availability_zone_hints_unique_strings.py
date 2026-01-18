from unittest import mock
from neutron_lib.api.validators import availability_zone as az_validator
from neutron_lib.db import constants as db_const
from neutron_lib import exceptions
from neutron_lib.tests import _base as base
@mock.patch.object(az_validator.validators, 'validate_list_of_unique_strings', return_value='bad')
def test__validate_availability_zone_hints_unique_strings(self, mock_unique_strs):
    self.assertEqual('bad', az_validator._validate_availability_zone_hints(['a', 'a']))
from unittest import mock
from neutron_lib.api.validators import availability_zone as az_validator
from neutron_lib.db import constants as db_const
from neutron_lib import exceptions
from neutron_lib.tests import _base as base
def test__validate_availability_zone_hints_excessive_len(self):
    self.assertRaisesRegex(exceptions.InvalidInput, 'Too many availability_zone_hints', az_validator._validate_availability_zone_hints, ['a' * (db_const.AZ_HINTS_DB_LEN + 1)])
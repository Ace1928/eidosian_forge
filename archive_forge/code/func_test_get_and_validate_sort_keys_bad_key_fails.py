from unittest import mock
from oslo_config import cfg
from oslo_db.sqlalchemy import models
import sqlalchemy as sa
from sqlalchemy.ext import declarative
from sqlalchemy import orm
from neutron_lib.api import attributes
from neutron_lib import context
from neutron_lib.db import utils
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
def test_get_and_validate_sort_keys_bad_key_fails(self):
    sorts = [('master', True)]
    self.assertRaises(n_exc.BadRequest, utils.get_and_validate_sort_keys, sorts, FakePort)
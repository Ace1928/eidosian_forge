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
def test_get_marker_obj_no_limit_and_marker(self):
    self.assertIsNone(utils.get_marker_obj(mock.Mock(), 'ctx', 'myr', 0, mock.ANY))
    self.assertIsNone(utils.get_marker_obj(mock.Mock(), 'ctx', 'myr', 10, None))
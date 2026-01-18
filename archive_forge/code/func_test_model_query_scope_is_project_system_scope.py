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
def test_model_query_scope_is_project_system_scope(self):
    ctx = context.Context(system_scope='all')
    model = mock.Mock(project_id='project')
    self.assertTrue(utils.model_query_scope_is_project(ctx, model))
    del model.project_id
    self.assertFalse(utils.model_query_scope_is_project(ctx, model))
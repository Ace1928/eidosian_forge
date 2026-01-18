import copy
import datetime
import functools
from unittest import mock
import uuid
from oslo_db import exception as db_exception
from oslo_db.sqlalchemy import utils as sqlalchemyutils
from sqlalchemy import sql
from glance.common import exception
from glance.common import timeutils
from glance import context
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import models
from glance.tests import functional
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def setup_contexts(self):
    self.admin_context = context.RequestContext(is_admin=True, tenant=self.admin_tenant)
    self.admin_none_context = context.RequestContext(is_admin=True, tenant=None)
    self.tenant1_context = context.RequestContext(tenant=self.tenant1)
    self.tenant2_context = context.RequestContext(tenant=self.tenant2)
    self.none_context = context.RequestContext(tenant=None)
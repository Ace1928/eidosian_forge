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
def build_image_fixtures(self):
    fixtures = []
    owners = {'Unowned': None, 'Admin Tenant': self.admin_tenant, 'Tenant 1': self.tenant1, 'Tenant 2': self.tenant2}
    visibilities = ['community', 'private', 'public', 'shared']
    for owner_label, owner in owners.items():
        for visibility in visibilities:
            fixture = {'name': '%s, %s' % (owner_label, visibility), 'owner': owner, 'visibility': visibility}
            fixtures.append(fixture)
    return [build_image_fixture(**f) for f in fixtures]
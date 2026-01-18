import datetime
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exc
from oslo_utils import encodeutils
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import timeutils
from sqlalchemy import orm as sa_orm
from glance.common import crypt
from glance.common import exception
import glance.context
import glance.db
from glance.db.sqlalchemy import api
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_list_with_type(self):
    filters = {'type': 'import'}
    tasks = self.task_repo.list(filters=filters)
    task_ids = set([i.task_id for i in tasks])
    self.assertEqual(set([UUID1, UUID2, UUID3]), task_ids)
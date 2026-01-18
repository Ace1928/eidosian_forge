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
def build_fixtures(self):
    dt1 = timeutils.utcnow() - datetime.timedelta(days=5)
    dt2 = dt1 + datetime.timedelta(days=1)
    dt3 = dt2 + datetime.timedelta(days=1)
    fixtures = [{'created_at': dt1, 'updated_at': dt1, 'deleted_at': dt3, 'deleted': True}, {'created_at': dt1, 'updated_at': dt2, 'deleted_at': timeutils.utcnow(), 'deleted': True}, {'created_at': dt2, 'updated_at': dt2, 'deleted_at': None, 'deleted': False}]
    return ([build_image_fixture(**fixture) for fixture in fixtures], [build_task_fixture(**fixture) for fixture in fixtures])
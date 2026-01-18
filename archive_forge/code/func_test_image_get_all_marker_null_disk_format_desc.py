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
def test_image_get_all_marker_null_disk_format_desc(self):
    """Check an image with disk_format null is handled

        Check an image with disk_format null is handled when
        marker is specified and order is descending
        """
    TENANT1 = str(uuid.uuid4())
    ctxt1 = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
    UUIDX = str(uuid.uuid4())
    self.db_api.image_create(ctxt1, {'id': UUIDX, 'status': 'queued', 'disk_format': None, 'owner': TENANT1})
    images = self.db_api.image_get_all(ctxt1, marker=UUIDX, sort_key=['disk_format'], sort_dir=['desc'])
    image_ids = [image['id'] for image in images]
    expected = []
    self.assertEqual(sorted(expected), sorted(image_ids))
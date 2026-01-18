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
def test_image_paginate(self):
    """Paginate through a list of images using limit and marker"""
    now = timeutils.utcnow()
    extra_uuids = [(str(uuid.uuid4()), now + datetime.timedelta(seconds=i * 5)) for i in range(2)]
    extra_images = [build_image_fixture(id=_id, created_at=_dt, updated_at=_dt) for _id, _dt in extra_uuids]
    self.create_images(extra_images)
    extra_uuids.reverse()
    page = self.db_api.image_get_all(self.context, limit=2)
    self.assertEqual([i[0] for i in extra_uuids], [i['id'] for i in page])
    last = page[-1]['id']
    page = self.db_api.image_get_all(self.context, limit=2, marker=last)
    self.assertEqual([UUID3, UUID2], [i['id'] for i in page])
    page = self.db_api.image_get_all(self.context, limit=2, marker=UUID2)
    self.assertEqual([UUID1], [i['id'] for i in page])
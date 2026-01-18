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
def test_image_get_all_with_filter_user_deleted_property(self):
    fixture = {'name': 'poo', 'value': 'bear', 'image_id': UUID1}
    prop = self.db_api.image_property_create(self.context, fixture)
    images = self.db_api.image_get_all(self.context, filters={'properties': {'poo': 'bear'}})
    self.assertEqual(1, len(images))
    self.db_api.image_property_delete(self.context, prop['name'], images[0]['id'])
    images = self.db_api.image_get_all(self.context, filters={'properties': {'poo': 'bear'}})
    self.assertEqual(0, len(images))
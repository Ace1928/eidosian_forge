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
def test_image_update_with_locations(self):
    locations = [{'url': 'a', 'metadata': {}, 'status': 'active'}, {'url': 'b', 'metadata': {}, 'status': 'active'}]
    fixture = {'locations': locations}
    image = self.db_api.image_update(self.adm_context, UUID3, fixture)
    self.assertEqual(2, len(image['locations']))
    self.assertIn('id', image['locations'][0])
    self.assertIn('id', image['locations'][1])
    image['locations'][0].pop('id')
    image['locations'][1].pop('id')
    self.assertEqual(locations, image['locations'])
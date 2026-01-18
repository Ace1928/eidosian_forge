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
def test_image_update_purge_properties(self):
    fixture = {'properties': {'ping': 'pong'}}
    image = self.db_api.image_update(self.adm_context, UUID1, fixture, purge_props=True)
    properties = {p['name']: p for p in image['properties']}
    self.assertIn('ping', properties)
    self.assertEqual('pong', properties['ping']['value'])
    self.assertFalse(properties['ping']['deleted'])
    self.assertIn('foo', properties)
    self.assertEqual('bar', properties['foo']['value'])
    self.assertTrue(properties['foo']['deleted'])
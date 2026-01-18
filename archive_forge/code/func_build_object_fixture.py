import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def build_object_fixture(**kwargs):
    object = {'namespace_id': 1, 'name': 'test-object-name', 'description': 'test-object-description', 'required': 'fake-required-properties-list', 'json_schema': '{fake-schema}'}
    object.update(kwargs)
    return object
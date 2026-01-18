import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def build_property_fixture(**kwargs):
    property = {'namespace_id': 1, 'name': 'test-property-name', 'json_schema': '{fake-schema}'}
    property.update(kwargs)
    return property
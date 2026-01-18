import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def build_namespace_fixture(**kwargs):
    namespace = {'namespace': 'MyTestNamespace', 'display_name': 'test-display-name', 'description': 'test-description', 'visibility': 'public', 'protected': 0, 'owner': 'test-owner'}
    namespace.update(kwargs)
    return namespace
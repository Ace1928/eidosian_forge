import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def test_association_delete(self):
    ns_fixture = build_namespace_fixture()
    ns_created = self.db_api.metadef_namespace_create(self.context, ns_fixture)
    self.assertIsNotNone(ns_created, 'Could not create a namespace.')
    self._assert_saved_fields(ns_fixture, ns_created)
    fixture = build_association_fixture()
    created = self.db_api.metadef_resource_type_association_create(self.context, ns_created['namespace'], fixture)
    self.assertIsNotNone(created, 'Could not create an association.')
    created_resource = self.db_api.metadef_resource_type_get(self.context, fixture['name'])
    self.assertIsNotNone(created_resource, 'resource_type not created')
    self.db_api.metadef_resource_type_association_delete(self.context, ns_created['namespace'], created_resource['name'])
    self.assertRaises(exception.NotFound, self.db_api.metadef_resource_type_association_get, self.context, ns_created['namespace'], created_resource['name'])
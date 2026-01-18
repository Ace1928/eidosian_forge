from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_add_property(self):
    property = _db_property_fixture(name='added_property')
    self.assertEqual('added_property', property['name'])
    self.db.metadef_property_create(self.context, NAMESPACE1, property)
    retrieved_property = self.property_repo.get(NAMESPACE1, 'added_property')
    self.assertEqual('added_property', retrieved_property.name)
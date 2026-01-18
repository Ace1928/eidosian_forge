from openstack.image.v2 import metadef_schema as _metadef_schema
from openstack.tests.functional.image.v2 import base
def test_get_metadef_tags_schema(self):
    metadef_schema = self.conn.image.get_metadef_tags_schema()
    self.assertIsNotNone(metadef_schema)
    self.assertIsInstance(metadef_schema, _metadef_schema.MetadefSchema)
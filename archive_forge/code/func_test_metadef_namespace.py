from openstack.image.v2 import metadef_namespace as _metadef_namespace
from openstack.tests.functional.image.v2 import base
def test_metadef_namespace(self):
    metadef_namespace = self.conn.image.get_metadef_namespace(self.metadef_namespace.namespace)
    self.assertEqual(self.metadef_namespace.namespace, metadef_namespace.namespace)
    metadef_namespaces = list(self.conn.image.metadef_namespaces())
    self.assertIn(self.metadef_namespace.namespace, {n.namespace for n in metadef_namespaces})
    metadef_namespace_display_name = 'A display name'
    metadef_namespace_description = 'A description'
    metadef_namespace = self.conn.image.update_metadef_namespace(self.metadef_namespace, display_name=metadef_namespace_display_name, description=metadef_namespace_description)
    self.assertIsInstance(metadef_namespace, _metadef_namespace.MetadefNamespace)
    metadef_namespace = self.conn.image.get_metadef_namespace(self.metadef_namespace.namespace)
    self.assertEqual(metadef_namespace_display_name, metadef_namespace.display_name)
    self.assertEqual(metadef_namespace_description, metadef_namespace.description)
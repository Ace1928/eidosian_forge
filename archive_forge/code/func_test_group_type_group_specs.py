from openstack.block_storage.v3 import group as _group
from openstack.block_storage.v3 import group_snapshot as _group_snapshot
from openstack.block_storage.v3 import group_type as _group_type
from openstack.block_storage.v3 import volume as _volume
from openstack.tests.functional.block_storage.v3 import base
def test_group_type_group_specs(self):
    group_type = self.conn.block_storage.create_group_type_group_specs(self.group_type, {'foo': 'bar', 'acme': 'buzz'})
    self.assertIsInstance(group_type, _group_type.GroupType)
    group_type = self.conn.block_storage.get_group_type(self.group_type.id)
    self.assertEqual({'foo': 'bar', 'acme': 'buzz'}, group_type.group_specs)
    spec = self.conn.block_storage.get_group_type_group_specs_property(self.group_type, 'foo')
    self.assertEqual('bar', spec)
    spec = self.conn.block_storage.update_group_type_group_specs_property(self.group_type, 'foo', 'baz')
    self.assertEqual('baz', spec)
    group_type = self.conn.block_storage.get_group_type(self.group_type.id)
    self.assertEqual({'foo': 'baz', 'acme': 'buzz'}, group_type.group_specs)
    self.conn.block_storage.delete_group_type_group_specs_property(self.group_type, 'foo')
    group_type = self.conn.block_storage.get_group_type(self.group_type.id)
    self.assertEqual({'acme': 'buzz'}, group_type.group_specs)
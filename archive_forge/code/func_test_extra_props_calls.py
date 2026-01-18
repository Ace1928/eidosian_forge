import uuid
from openstack import exceptions
from openstack.tests.functional import base
def test_extra_props_calls(self):
    flavor_name = uuid.uuid4().hex
    flv = self.conn.compute.create_flavor(is_public=False, name=flavor_name, ram=128, vcpus=1, disk=0)
    self.addCleanup(self.conn.compute.delete_flavor, flv.id)
    specs = {'a': 'b'}
    self.conn.compute.create_flavor_extra_specs(flv, extra_specs=specs)
    flv_cmp = self.conn.compute.fetch_flavor_extra_specs(flv)
    self.assertDictEqual(specs, flv_cmp.extra_specs)
    self.conn.compute.update_flavor_extra_specs_property(flv, 'c', 'd')
    val_cmp = self.conn.compute.get_flavor_extra_specs_property(flv, 'c')
    self.assertEqual('d', val_cmp)
    self.conn.compute.delete_flavor_extra_specs_property(flv, 'c')
    flv_cmp = self.conn.compute.fetch_flavor_extra_specs(flv)
    self.assertDictEqual(specs, flv_cmp.extra_specs)
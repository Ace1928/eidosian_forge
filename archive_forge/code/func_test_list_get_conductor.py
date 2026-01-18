from openstack.tests.functional.baremetal import base
def test_list_get_conductor(self):
    node = self.create_node(name='node-name')
    conductors = self.conn.baremetal.conductors()
    hostname_list = [conductor.hostname for conductor in conductors]
    self.assertIn(node.conductor, hostname_list)
    conductor1 = self.conn.baremetal.get_conductor(node.conductor)
    self.assertIsNotNone(conductor1.conductor_group)
    self.assertIsNotNone(conductor1.links)
    self.assertTrue(conductor1.alive)
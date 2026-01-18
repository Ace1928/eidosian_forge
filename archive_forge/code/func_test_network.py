from examples import connect
from examples.network import create as network_create
from examples.network import delete as network_delete
from examples.network import find as network_find
from examples.network import list as network_list
from openstack.tests.functional import base
def test_network(self):
    network_list.list_networks(self.conn)
    network_list.list_subnets(self.conn)
    network_list.list_ports(self.conn)
    network_list.list_security_groups(self.conn)
    network_list.list_routers(self.conn)
    network_find.find_network(self.conn)
    network_create.create_network(self.conn)
    network_delete.delete_network(self.conn)
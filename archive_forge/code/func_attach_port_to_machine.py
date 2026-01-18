import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def attach_port_to_machine(self, name_or_id, port_name_or_id):
    """Attach a virtual port to the bare metal machine.

        :param string name_or_id: A machine name or UUID.
        :param string port_name_or_id: A port name or UUID.
            Note that this is a Network service port, not a bare metal NIC.
        :return: Nothing.
        """
    machine = self.get_machine(name_or_id)
    port = self.get_port(port_name_or_id)
    self.baremetal.attach_vif_to_node(machine, port['id'])
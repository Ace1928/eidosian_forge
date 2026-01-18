import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def set_machine_power_on(self, name_or_id):
    """Activate baremetal machine power

        This is a method that sets the node power state to "on".

        :params string name_or_id: A string representing the baremetal
            node to have power turned to an "on" state.

        :returns: None
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    self.baremetal.set_node_power_state(name_or_id, 'power on')
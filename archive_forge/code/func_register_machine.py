import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def register_machine(self, nics, wait=False, timeout=3600, lock_timeout=600, provision_state='available', **kwargs):
    """Register Baremetal with Ironic

        Allows for the registration of Baremetal nodes with Ironic
        and population of pertinant node information or configuration
        to be passed to the Ironic API for the node.

        This method also creates ports for a list of MAC addresses passed
        in to be utilized for boot and potentially network configuration.

        If a failure is detected creating the network ports, any ports
        created are deleted, and the node is removed from Ironic.

        :param nics:
            An array of ports that represent the network interfaces for the
            node to be created. The ports are created after the node is
            enrolled but before it goes through cleaning.

            Example::

                [
                    {'address': 'aa:bb:cc:dd:ee:01'},
                    {'address': 'aa:bb:cc:dd:ee:02'}
                ]

            Alternatively, you can provide an array of MAC addresses.
        :param wait: Boolean value, defaulting to false, to wait for the node
            to reach the available state where the node can be provisioned. It
            must be noted, when set to false, the method will still wait for
            locks to clear before sending the next required command.
        :param timeout: Integer value, defautling to 3600 seconds, for the wait
            state to reach completion.
        :param lock_timeout: Integer value, defaulting to 600 seconds, for
            locks to clear.
        :param provision_state: The expected provision state, one of "enroll"
            "manageable" or "available". Using "available" results in automated
            cleaning.
        :param kwargs: Key value pairs to be passed to the Ironic API,
            including uuid, name, chassis_uuid, driver_info, properties.

        :returns: Current state of the node.
        :rtype: :class:`~openstack.baremetal.v1.node.Node`.
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    if provision_state not in ('enroll', 'manageable', 'available'):
        raise ValueError('Initial provision state must be enroll, manageable or available, got %s' % provision_state)
    if provision_state != 'available':
        kwargs['provision_state'] = 'enroll'
    machine = self.baremetal.create_node(**kwargs)
    with self._delete_node_on_error(machine):
        if machine.provision_state == 'enroll' and provision_state != 'enroll':
            machine = self.baremetal.set_node_provision_state(machine, 'manage', wait=True, timeout=timeout)
            machine = self.baremetal.wait_for_node_reservation(machine, timeout=lock_timeout)
        created_nics = []
        try:
            for port in _normalize_port_list(nics):
                nic = self.baremetal.create_port(node_id=machine.id, **port)
                created_nics.append(nic.id)
        except Exception:
            for uuid in created_nics:
                try:
                    self.baremetal.delete_port(uuid)
                except Exception:
                    pass
            raise
        if machine.provision_state != 'available' and provision_state == 'available':
            machine = self.baremetal.set_node_provision_state(machine, 'provide', wait=wait, timeout=timeout)
        return machine
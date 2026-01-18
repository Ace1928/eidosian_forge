import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def node_set_provision_state(self, name_or_id, state, configdrive=None, wait=False, timeout=3600):
    """Set Node Provision State

        Enables a user to provision a Machine and optionally define a
        config drive to be utilized.

        :param string name_or_id: The Name or UUID value representing the
            baremetal node.
        :param string state: The desired provision state for the baremetal
            node.
        :param string configdrive: An optional URL or file or path
            representing the configdrive. In the case of a directory, the
            client API will create a properly formatted configuration drive
            file and post the file contents to the API for deployment.
        :param boolean wait: A boolean value, defaulted to false, to control
            if the method will wait for the desire end state to be reached
            before returning.
        :param integer timeout: Integer value, defaulting to 3600 seconds,
            representing the amount of time to wait for the desire end state to
            be reached.

        :returns: Current state of the machine upon exit of the method.
        :rtype: :class:`~openstack.baremetal.v1.node.Node`.
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    node = self.baremetal.set_node_provision_state(name_or_id, target=state, config_drive=configdrive, wait=wait, timeout=timeout)
    return node
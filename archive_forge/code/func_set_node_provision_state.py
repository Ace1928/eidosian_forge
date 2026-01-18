from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import allocation as _allocation
from openstack.baremetal.v1 import chassis as _chassis
from openstack.baremetal.v1 import conductor as _conductor
from openstack.baremetal.v1 import deploy_templates as _deploytemplates
from openstack.baremetal.v1 import driver as _driver
from openstack.baremetal.v1 import node as _node
from openstack.baremetal.v1 import port as _port
from openstack.baremetal.v1 import port_group as _portgroup
from openstack.baremetal.v1 import volume_connector as _volumeconnector
from openstack.baremetal.v1 import volume_target as _volumetarget
from openstack import exceptions
from openstack import proxy
from openstack import utils
def set_node_provision_state(self, node, target, config_drive=None, clean_steps=None, rescue_password=None, wait=False, timeout=None, deploy_steps=None):
    """Run an action modifying node's provision state.

        This call is asynchronous, it will return success as soon as the Bare
        Metal service acknowledges the request.

        :param node: The value can be the name or ID of a node or a
            :class:`~openstack.baremetal.v1.node.Node` instance.
        :param target: Provisioning action, e.g. ``active``, ``provide``.
            See the Bare Metal service documentation for available actions.
        :param config_drive: Config drive to pass to the node, only valid
            for ``active` and ``rebuild`` targets. You can use functions from
            :mod:`openstack.baremetal.configdrive` to build it.
        :param clean_steps: Clean steps to execute, only valid for ``clean``
            target.
        :param rescue_password: Password for the rescue operation, only valid
            for ``rescue`` target.
        :param wait: Whether to wait for the node to get into the expected
            state. The expected state is determined from a combination of
            the current provision state and ``target``.
        :param timeout: If ``wait`` is set to ``True``, specifies how much (in
            seconds) to wait for the expected state to be reached. The value of
            ``None`` (the default) means no client-side timeout.
        :param deploy_steps: Deploy steps to execute, only valid for ``active``
            and ``rebuild`` target.

        :returns: The updated :class:`~openstack.baremetal.v1.node.Node`
        :raises: ValueError if ``config_drive``, ``clean_steps``,
            ``deploy_steps`` or ``rescue_password`` are provided with an
            invalid ``target``.
        """
    res = self._get_resource(_node.Node, node)
    return res.set_provision_state(self, target, config_drive=config_drive, clean_steps=clean_steps, rescue_password=rescue_password, wait=wait, timeout=timeout, deploy_steps=deploy_steps)
from libcloud.utils.py3 import ET, tostring
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.common.abiquo import AbiquoResponse, AbiquoConnection, get_href
from libcloud.compute.types import Provider, LibcloudError
def ex_run_node(self, node):
    """
        Runs a node

        Here there is a bit difference between Abiquo states and libcloud
        states, so this method is created to have better compatibility. In
        libcloud, if the node is not running, then it does not exist (avoiding
        UNKNOWN and temporal states). In Abiquo, you can define a node, and
        then deploy it.

        If the node is in :class:`NodeState.TERMINATED` libcloud's state and in
        'NOT_DEPLOYED' Abiquo state, there is a way to run and recover it
        for libcloud using this method. There is no way to reach this state
        if you are using only libcloud, but you may have used another Abiquo
        client and now you want to recover your node to be used by libcloud.

        :param node: The node to run
        :type node: :class:`Node`

        :return: The node itself, but with the new state
        :rtype: :class:`Node`
        """
    e_vm = self.connection.request(node.extra['uri_id']).object
    state = e_vm.findtext('state')
    if state != 'NOT_ALLOCATED':
        raise LibcloudError('Invalid Node state', self)
    self._deploy_remote(e_vm)
    edit_vm = get_href(e_vm, 'edit')
    headers = {'Accept': self.NODE_MIME_TYPE}
    e_vm = self.connection.request(edit_vm, headers=headers).object
    return self._to_node(e_vm, self)
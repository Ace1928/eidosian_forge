from libcloud.utils.py3 import ET, tostring
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.common.abiquo import AbiquoResponse, AbiquoConnection, get_href
from libcloud.compute.types import Provider, LibcloudError
def ex_destroy_group(self, group):
    """
        Destroy a group.

        Be careful! Destroying a group means destroying all the :class:`Node`
        instances there and the group itself!

        If there is currently any action over any :class:`Node` of the
        :class:`NodeGroup`, then the method will raise an exception.

        :param     name: The group (required)
        :type      name: :class:`NodeGroup`

        :return:         If the group was destroyed successfully
        :rtype:          ``bool``
        """
    e_group = self.connection.request(group.uri).object
    state = e_group.findtext('state')
    if state not in ['NOT_DEPLOYED', 'DEPLOYED']:
        error = 'Can not destroy group because of current state'
        raise LibcloudError(error, self)
    if state == 'DEPLOYED':
        vm_task = ET.Element('virtualmachinetask')
        force_undeploy = ET.SubElement(vm_task, 'forceUndeploy')
        force_undeploy.text = 'True'
        undeploy_uri = group.uri + '/action/undeploy'
        headers = {'Accept': self.AR_MIME_TYPE, 'Content-type': self.VM_TASK_MIME_TYPE}
        res = self.connection.async_request(action=undeploy_uri, method='POST', data=tostring(vm_task), headers=headers)
    if state == 'NOT_DEPLOYED' or res.async_success():
        self.connection.request(action=group.uri, method='DELETE')
        return True
    else:
        return False
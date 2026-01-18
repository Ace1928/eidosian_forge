import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def set_boot_device(self, session, boot_device, persistent=False):
    """Set node boot device

        :param session: The session to use for making this request.
        :param boot_device: Boot device to assign to the node.
        :param persistent: If the boot device change is maintained after node
            reboot
        :returns: ``None``
        """
    session = self._get_session(session)
    version = self._get_microversion(session, action='commit')
    request = self._prepare_request(requires_id=True)
    request.url = utils.urljoin(request.url, 'management', 'boot_device')
    body = {'boot_device': boot_device, 'persistent': persistent}
    response = session.put(request.url, json=body, headers=request.headers, microversion=version, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)
    msg = 'Failed to set boot device for node {node}'.format(node=self.id)
    exceptions.raise_from_response(response, error_message=msg)
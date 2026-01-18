import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def detach_vif(self, session, vif_id, ignore_missing=True):
    """Detach a VIF from the node.

        The exact form of the VIF ID depends on the network interface used by
        the node. In the most common case it is a Network service port
        (NOT a Bare Metal port) ID.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param string vif_id: Backend-specific VIF ID.
        :param bool ignore_missing: When set to ``False``
                    :class:`~openstack.exceptions.ResourceNotFound` will be
                    raised when the VIF does not exist. Otherwise, ``False``
                    is returned.
        :return: ``True`` if the VIF was detached, otherwise ``False``.
        :raises: :exc:`~openstack.exceptions.NotSupported` if the server
            does not support the VIF API.
        """
    session = self._get_session(session)
    version = self._assert_microversion_for(session, 'commit', _common.VIF_VERSION, error_message='Cannot use VIF attachment API')
    request = self._prepare_request(requires_id=True)
    request.url = utils.urljoin(request.url, 'vifs', vif_id)
    response = session.delete(request.url, headers=request.headers, microversion=version, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)
    if ignore_missing and response.status_code == 400:
        session.log.debug('VIF %(vif)s was already removed from node %(node)s', {'vif': vif_id, 'node': self.id})
        return False
    msg = 'Failed to detach VIF {vif} from bare metal node {node}'.format(node=self.id, vif=vif_id)
    exceptions.raise_from_response(response, error_message=msg)
    return True
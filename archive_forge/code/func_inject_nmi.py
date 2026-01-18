import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def inject_nmi(self, session):
    """Inject NMI.

        :param session: The session to use for making this request.
        :return: None
        """
    session = self._get_session(session)
    version = self._assert_microversion_for(session, 'commit', _common.INJECT_NMI_VERSION)
    request = self._prepare_request(requires_id=True)
    request.url = utils.urljoin(request.url, 'management', 'inject_nmi')
    response = session.put(request.url, json={}, headers=request.headers, microversion=version, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)
    msg = 'Failed to inject NMI to node {node}'.format(node=self.id)
    exceptions.raise_from_response(response, error_message=msg)
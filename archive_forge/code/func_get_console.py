import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def get_console(self, session):
    """Get the node console.

        :param session: The session to use for making this request.
        :returns: The HTTP response.
        """
    session = self._get_session(session)
    version = self._get_microversion(session, action='fetch')
    request = self._prepare_request(requires_id=True)
    request.url = utils.urljoin(request.url, 'states', 'console')
    response = session.get(request.url, headers=request.headers, microversion=version, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)
    msg = 'Failed to get console for node {node}'.format(node=self.id)
    exceptions.raise_from_response(response, error_message=msg)
    return response.json()
import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def set_traits(self, session, traits):
    """Set traits for the node.

        Removes any existing traits and adds the traits passed in to this
        method.

        :param session: The session to use for making this request.
        :param traits: list of traits to add to the node.
        :returns: ``None``
        """
    session = self._get_session(session)
    version = utils.pick_microversion(session, '1.37')
    request = self._prepare_request(requires_id=True)
    request.url = utils.urljoin(request.url, 'traits')
    body = {'traits': traits}
    response = session.put(request.url, json=body, headers=request.headers, microversion=version, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)
    msg = 'Failed to set traits for node {node}'.format(node=self.id)
    exceptions.raise_from_response(response, error_message=msg)
    self.traits = traits
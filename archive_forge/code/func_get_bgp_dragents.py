from openstack import exceptions
from openstack import resource
from openstack import utils
def get_bgp_dragents(self, session):
    """List Dynamic Routing Agents hosting a specific BGP Speaker

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :returns: The response as a list of dragents hosting a specific
                  BGP Speaker.

        :raises: :class:`~openstack.exceptions.SDKException` on error.
        """
    url = utils.urljoin(self.base_path, self.id, 'bgp-dragents')
    resp = session.get(url)
    exceptions.raise_from_response(resp)
    self._body.attributes.update(resp.json())
    return resp.json()
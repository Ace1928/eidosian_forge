from openstack import exceptions
from openstack import resource
from openstack import utils
def get_advertised_routes(self, session):
    """List routes advertised by a BGP Speaker

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :returns: The response as a list of routes (cidr/nexthop pair
                  advertised by the BGP Speaker.

        :raises: :class:`~openstack.exceptions.SDKException` on error.
        """
    url = utils.urljoin(self.base_path, self.id, 'get_advertised_routes')
    resp = session.get(url)
    exceptions.raise_from_response(resp)
    self._body.attributes.update(resp.json())
    return resp.json()
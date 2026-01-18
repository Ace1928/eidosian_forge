from openstack import exceptions
from openstack import resource
from openstack import utils
def get_bgp_speakers_hosted_by_dragent(self, session):
    """List BGP speakers hosted by a Dynamic Routing Agent

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        """
    url = utils.urljoin(self.base_path, self.id, 'bgp-drinstances')
    resp = session.get(url)
    exceptions.raise_from_response(resp)
    self._body.attributes.update(resp.json())
    return resp.json()
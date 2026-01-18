import typing as ty
from openstack.common import metadata
from openstack.common import tag
from openstack.compute.v2 import flavor
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.image.v2 import image
from openstack import resource
from openstack import utils
def fetch_topology(self, session):
    """Fetch the topology information for the server.

        :param session: The session to use for making this request.
        :returns: None
        """
    utils.require_microversion(session, 2.78)
    url = utils.urljoin(Server.base_path, self.id, 'topology')
    response = session.get(url)
    exceptions.raise_from_response(response)
    try:
        return response.json()
    except ValueError:
        pass
import typing as ty
from openstack.common import metadata
from openstack.common import tag
from openstack.compute.v2 import flavor
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.image.v2 import image
from openstack import resource
from openstack import utils
def fetch_security_groups(self, session):
    """Fetch security groups of the server.

        :param session: The session to use for making this request.
        :returns: Updated Server instance.
        """
    url = utils.urljoin(Server.base_path, self.id, 'os-security-groups')
    response = session.get(url)
    exceptions.raise_from_response(response)
    try:
        data = response.json()
        if 'security_groups' in data:
            self.security_groups = data['security_groups']
    except ValueError:
        pass
    return self
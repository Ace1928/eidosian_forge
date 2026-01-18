from openstack import exceptions
from openstack import resource
from openstack import utils
def delete_all_properties(self, session):
    """Delete all properties in a namespace.

        :param session: The session to use for making this request
        :returns: The server response
        """
    url = utils.urljoin(self.base_path, self.id, 'properties')
    return self._delete_all(session, url)
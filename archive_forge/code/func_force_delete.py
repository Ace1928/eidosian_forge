from openstack.clustering.v1 import _async_resource
from openstack import resource
from openstack import utils
def force_delete(self, session):
    """Force delete a node."""
    body = {'force': True}
    url = utils.urljoin(self.base_path, self.id)
    response = session.delete(url, json=body)
    return self._delete_response(response)
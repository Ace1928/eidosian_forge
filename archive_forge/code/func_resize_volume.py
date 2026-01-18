from openstack import resource
from openstack import utils
def resize_volume(self, session, volume_size):
    """Resize the volume attached to the instance

        :returns: ``None``
        """
    body = {'resize': {'volume': volume_size}}
    url = utils.urljoin(self.base_path, self.id, 'action')
    session.post(url, json=body)
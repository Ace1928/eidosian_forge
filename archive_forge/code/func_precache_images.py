from openstack import exceptions
from openstack import resource
from openstack import utils
def precache_images(self, session, images):
    """Requests image pre-caching"""
    body = {'cache': images}
    url = utils.urljoin(self.base_path, self.id, 'images')
    response = session.post(url, json=body, microversion=self._max_microversion)
    exceptions.raise_from_response(response)
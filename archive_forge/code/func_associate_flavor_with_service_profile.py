from openstack import resource
from openstack import utils
def associate_flavor_with_service_profile(self, session, service_profile_id=None):
    flavor_id = self.id
    url = utils.urljoin(self.base_path, flavor_id, 'service_profiles')
    body = {'service_profile': {'id': service_profile_id}}
    resp = session.post(url, json=body)
    return resp.json()
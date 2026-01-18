from openstack.common import tag
from openstack import resource
from openstack import utils
def assign_role_to_group(self, session, group, role):
    """Assign role to group on project"""
    url = utils.urljoin(self.base_path, self.id, 'groups', group.id, 'roles', role.id)
    resp = session.put(url)
    if resp.status_code == 204:
        return True
    return False
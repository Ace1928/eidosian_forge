from openstack import exceptions
from openstack import resource
from openstack import utils
def create_group_specs(self, session, specs):
    """Creates group specs for the group type.

        This will override whatever specs are already present on the group
        type.

        :param session: The session to use for making this request.
        :param specs: A dict of group specs to set on the group type.
        :returns: An updated version of this object.
        """
    url = utils.urljoin(GroupType.base_path, self.id, 'group_specs')
    microversion = self._get_microversion(session, action='create')
    response = session.post(url, json={'group_specs': specs}, microversion=microversion)
    exceptions.raise_from_response(response)
    specs = response.json().get('group_specs', {})
    self._update(group_specs=specs)
    return self
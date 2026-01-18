from openstack import exceptions
from openstack import resource
from openstack import utils
def delete_group_specs_property(self, session, prop):
    """Delete a group spec property from the group type.

        :param session: The session to use for making this request.
        :param prop: The name of the group spec property to delete.
        :returns: None
        """
    url = utils.urljoin(GroupType.base_path, self.id, 'group_specs', prop)
    microversion = self._get_microversion(session, action='delete')
    response = session.delete(url, microversion=microversion)
    exceptions.raise_from_response(response)
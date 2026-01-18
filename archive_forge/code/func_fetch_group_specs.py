from openstack import exceptions
from openstack import resource
from openstack import utils
def fetch_group_specs(self, session):
    """Fetch group_specs of the group type.

        These are returned by default if the user has suitable permissions
        (i.e. you're an admin) but by default you also need the same
        permissions to access this API. That means this function is kind of
        useless. However, that is how the API was designed and it is
        theoretically possible that people will have modified their policy to
        allow this but not the other so we provide this anyway.

        :param session: The session to use for making this request.
        :returns: An updated version of this object.
        """
    url = utils.urljoin(GroupType.base_path, self.id, 'group_specs')
    microversion = self._get_microversion(session, action='fetch')
    response = session.get(url, microversion=microversion)
    exceptions.raise_from_response(response)
    specs = response.json().get('group_specs', {})
    self._update(group_specs=specs)
    return self
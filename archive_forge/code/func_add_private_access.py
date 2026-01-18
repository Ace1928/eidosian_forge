from openstack import exceptions
from openstack import resource
from openstack import utils
def add_private_access(self, session, project_id):
    """Add project access from the volume type.

        :param session: The session to use for making this request.
        :param project_id: The project to add access for.
        """
    url = utils.urljoin(self.base_path, self.id, 'action')
    body = {'addProjectAccess': {'project': project_id}}
    resp = session.post(url, json=body)
    exceptions.raise_from_response(resp)
from openstack import resource
from openstack import utils
def enable_root_user(self, session):
    """Enable login for the root user.

        This operation enables login from any host for the root user
        and provides the user with a generated root password.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :returns: A dictionary with keys ``name`` and ``password`` specifying
            the login credentials.
        """
    url = utils.urljoin(self.base_path, self.id, 'root')
    resp = session.post(url)
    return resp.json()['user']
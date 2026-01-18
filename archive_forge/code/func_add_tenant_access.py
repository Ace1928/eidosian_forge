from openstack import exceptions
from openstack import resource
from openstack import utils
def add_tenant_access(self, session, tenant):
    """Adds flavor access to a tenant and flavor.

        :param session: The session to use for making this request.
        :param tenant:
        :returns: None
        """
    body = {'addTenantAccess': {'tenant': tenant}}
    self._action(session, body)
from openstack.identity.v2 import extension as _extension
from openstack.identity.v2 import role as _role
from openstack.identity.v2 import tenant as _tenant
from openstack.identity.v2 import user as _user
from openstack import proxy
def get_tenant(self, tenant):
    """Get a single tenant

        :param tenant: The value can be the ID of a tenant or a
            :class:`~openstack.identity.v2.tenant.Tenant` instance.

        :returns: One :class:`~openstack.identity.v2.tenant.Tenant`
        :raises: :class:`~openstack.exceptions.ResourceNotFound`
            when no resource can be found.
        """
    return self._get(_tenant.Tenant, tenant)
from openstack.identity.v2 import extension as _extension
from openstack.identity.v2 import role as _role
from openstack.identity.v2 import tenant as _tenant
from openstack.identity.v2 import user as _user
from openstack import proxy
def delete_tenant(self, tenant, ignore_missing=True):
    """Delete a tenant

        :param tenant: The value can be either the ID of a tenant or a
            :class:`~openstack.identity.v2.tenant.Tenant` instance.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the tenant does not exist.
            When set to ``True``, no exception will be set when
            attempting to delete a nonexistent tenant.

        :returns: ``None``
        """
    self._delete(_tenant.Tenant, tenant, ignore_missing=ignore_missing)
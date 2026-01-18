from openstack.identity.v2 import extension as _extension
from openstack.identity.v2 import role as _role
from openstack.identity.v2 import tenant as _tenant
from openstack.identity.v2 import user as _user
from openstack import proxy
def find_role(self, name_or_id, ignore_missing=True):
    """Find a single role

        :param name_or_id: The name or ID of a role.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the resource does not exist.
            When set to ``True``, None will be returned when
            attempting to find a nonexistent resource.
        :returns: One :class:`~openstack.identity.v2.role.Role` or None
        """
    return self._find(_role.Role, name_or_id, ignore_missing=ignore_missing)
from keystoneclient import base
def remove_user_role(self, user, role, tenant=None):
    """Remove a role from a user.

        If tenant is specified, the role is removed just for that tenant,
        otherwise the role is removed from the user's global roles.
        """
    user_id = base.getid(user)
    role_id = base.getid(role)
    if tenant:
        route = '/tenants/%s/users/%s/roles/OS-KSADM/%s'
        params = (base.getid(tenant), user_id, role_id)
        return self._delete(route % params)
    else:
        route = '/users/%s/roles/OS-KSADM/%s'
        return self._delete(route % (user_id, role_id))
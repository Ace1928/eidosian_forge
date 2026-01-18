from keystoneclient import base
def roles_for_user(self, user, tenant=None):
    user_id = base.getid(user)
    if tenant:
        tenant_id = base.getid(tenant)
        route = '/tenants/%s/users/%s/roles'
        return self._list(route % (tenant_id, user_id), 'roles')
    else:
        return self._list('/users/%s/roles' % user_id, 'roles')
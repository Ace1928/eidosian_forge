from troveclient import base
def get_by_tenant(self, datastore, tenant, limit=None, marker=None):
    """List members by tenant id."""
    return self._list('/mgmt/datastores/%s/versions/members/%s' % (datastore, tenant), 'datastore_version_members', limit, marker)
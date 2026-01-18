from openstack.database.v1 import database as _database
from openstack.database.v1 import flavor as _flavor
from openstack.database.v1 import instance as _instance
from openstack.database.v1 import user as _user
from openstack import proxy
def databases(self, instance, **query):
    """Return a generator of databases

        :param instance: This can be either the ID of an instance
            or a :class:`~openstack.database.v1.instance.Instance`
            instance that the interface belongs to.
        :param kwargs query: Optional query parameters to be sent to limit
            the resources being returned.

        :returns: A generator of database objects
        :rtype: :class:`~openstack.database.v1.database.Database`
        """
    instance = self._get_resource(_instance.Instance, instance)
    return self._list(_database.Database, instance_id=instance.id, **query)
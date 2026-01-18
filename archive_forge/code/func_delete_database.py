from openstack.database.v1 import database as _database
from openstack.database.v1 import flavor as _flavor
from openstack.database.v1 import instance as _instance
from openstack.database.v1 import user as _user
from openstack import proxy
def delete_database(self, database, instance=None, ignore_missing=True):
    """Delete a database

        :param database: The value can be either the ID of a database or a
            :class:`~openstack.database.v1.database.Database` instance.
        :param instance: This parameter needs to be specified when
            an ID is given as `database`.
            It can be either the ID of an instance
            or a :class:`~openstack.database.v1.instance.Instance`
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the database does not exist.
            When set to ``True``, no exception will be set when
            attempting to delete a nonexistent database.

        :returns: ``None``
        """
    instance_id = self._get_uri_attribute(database, instance, 'instance_id')
    self._delete(_database.Database, database, instance_id=instance_id, ignore_missing=ignore_missing)
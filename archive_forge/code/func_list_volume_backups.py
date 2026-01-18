import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def list_volume_backups(self, detailed=True, filters=None):
    """List all volume backups.

        :param detailed: Whether or not to add detailed additional information.
        :param filters: A dictionary of meta data to use for further filtering.
            Example::

                {
                    'name': 'my-volume-backup',
                    'status': 'available',
                    'volume_id': 'e126044c-7b4c-43be-a32a-c9cbbc9ddb56',
                    'all_tenants': 1
                }

        :returns: A list of volume ``Backup`` objects.
        """
    if not filters:
        filters = {}
    return list(self.block_storage.backups(details=detailed, **filters))
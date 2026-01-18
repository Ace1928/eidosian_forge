import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def search_volume_backups(self, name_or_id=None, filters=None):
    """Search for one or more volume backups.

        :param name_or_id: Name or unique ID of volume backup(s).
        :param filters: **DEPRECATED** A dictionary of meta data to use for
            further filtering. Elements of this dictionary may, themselves, be
            dictionaries. Example::

                {
                  'last_name': 'Smith',
                  'other': {
                      'gender': 'Female'
                  }
                }

            OR

            A string containing a jmespath expression for further filtering.
            Example::

                "[?last_name==`Smith`] | [?other.gender]==`Female`]"

        :returns: A list of volume ``Backup`` objects, if any are found.
        """
    volume_backups = self.list_volume_backups()
    return _utils._filter_list(volume_backups, name_or_id, filters)
import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def search_volume_snapshots(self, name_or_id=None, filters=None):
    """Search for one or more volume snapshots.

        :param name_or_id: Name or unique ID of volume snapshot(s).
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

        :returns: A list of volume ``Snapshot`` objects, if any are found.
        """
    volumesnapshots = self.list_volume_snapshots()
    return _utils._filter_list(volumesnapshots, name_or_id, filters)
import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def get_volume_type(self, name_or_id, filters=None):
    """Get a volume type by name or ID.

        :param name_or_id: Name or unique ID of the volume type.
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

        :returns: A volume ``Type`` object if found, else None.
        """
    return _utils._get_entity(self, 'volume_type', name_or_id, filters)
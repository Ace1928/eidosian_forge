import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def get_volume_type_access(self, name_or_id):
    """Return a list of volume_type_access.

        :param name_or_id: Name or unique ID of the volume type.
        :returns: A volume ``Type`` object if found, else None.
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    volume_type = self.get_volume_type(name_or_id)
    if not volume_type:
        raise exceptions.SDKException('VolumeType not found: %s' % name_or_id)
    return self.block_storage.get_type_access(volume_type)
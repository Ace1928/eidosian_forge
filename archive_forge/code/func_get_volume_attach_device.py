import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def get_volume_attach_device(self, volume, server_id):
    """Return the device name a volume is attached to for a server.

        This can also be used to verify if a volume is attached to
        a particular server.

        :param volume: The volume to fetch the device name from.
        :param server_id: ID of server to check.
        :returns: Device name if attached, None if volume is not attached.
        """
    for attach in volume['attachments']:
        if server_id == attach['server_id']:
            return attach['device']
    return None
import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def add_volume_type_access(self, name_or_id, project_id):
    """Grant access on a volume_type to a project.

        NOTE: the call works even if the project does not exist.

        :param name_or_id: ID or name of a volume_type
        :param project_id: A project id

        :returns: None
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    volume_type = self.get_volume_type(name_or_id)
    if not volume_type:
        raise exceptions.SDKException('VolumeType not found: %s' % name_or_id)
    self.block_storage.add_type_access(volume_type, project_id)
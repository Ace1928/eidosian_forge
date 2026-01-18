import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def set_volume_quotas(self, name_or_id, **kwargs):
    """Set a volume quota in a project

        :param name_or_id: project name or id
        :param kwargs: key/value pairs of quota name and quota value

        :returns: None
        :raises: :class:`~openstack.exceptions.SDKException` if the resource to
            set the quota does not exist.
        """
    proj = self.identity.find_project(name_or_id, ignore_missing=False)
    self.block_storage.update_quota_set(_qs.QuotaSet(project_id=proj.id), **kwargs)
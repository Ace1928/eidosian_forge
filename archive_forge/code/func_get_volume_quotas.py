import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def get_volume_quotas(self, name_or_id):
    """Get volume quotas for a project

        :param name_or_id: project name or id

        :returns: A volume ``QuotaSet`` object with the quotas
        :raises: :class:`~openstack.exceptions.SDKException` if it's not a
            valid project
        """
    proj = self.identity.find_project(name_or_id, ignore_missing=False)
    return self.block_storage.get_quota_set(proj)
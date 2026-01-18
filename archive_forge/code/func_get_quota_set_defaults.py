from openstack.block_storage import _base_proxy
from openstack.block_storage.v2 import backup as _backup
from openstack.block_storage.v2 import capabilities as _capabilities
from openstack.block_storage.v2 import extension as _extension
from openstack.block_storage.v2 import limits as _limits
from openstack.block_storage.v2 import quota_set as _quota_set
from openstack.block_storage.v2 import snapshot as _snapshot
from openstack.block_storage.v2 import stats as _stats
from openstack.block_storage.v2 import type as _type
from openstack.block_storage.v2 import volume as _volume
from openstack.identity.v3 import project as _project
from openstack import resource
def get_quota_set_defaults(self, project):
    """Show QuotaSet defaults for the project

        :param project: ID or instance of
            :class:`~openstack.identity.project.Project` of the project for
            which the quota should be retrieved

        :returns: One :class:`~openstack.block_storage.v2.quota_set.QuotaSet`
        :raises: :class:`~openstack.exceptions.ResourceNotFound`
            when no resource can be found.
        """
    project = self._get_resource(_project.Project, project)
    res = self._get_resource(_quota_set.QuotaSet, None, project_id=project.id)
    return res.fetch(self, base_path='/os-quota-sets/defaults')
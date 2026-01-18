from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def get_network_quotas(self, name_or_id, details=False):
    """Get network quotas for a project

        :param name_or_id: project name or id
        :param details: if set to True it will return details about usage
            of quotas by given project

        :returns: A network ``Quota`` object if found, else None.
        :raises: :class:`~openstack.exceptions.SDKException` if it's not a
            valid project
        """
    proj = self.identity.find_project(name_or_id, ignore_missing=False)
    return self.network.get_quota(proj.id, details)
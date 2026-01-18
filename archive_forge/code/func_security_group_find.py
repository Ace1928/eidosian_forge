from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
def security_group_find(self, security_group=None):
    """Return a security group given name or ID

        https://docs.openstack.org/api-ref/compute/#show-security-group-details

        :param string security_group:
            Security group name or ID
        :returns: A dict of the security group attributes
        """
    url = '/os-security-groups'
    return self.find(url, attr='name', value=security_group)
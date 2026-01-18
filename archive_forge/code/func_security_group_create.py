from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
def security_group_create(self, name=None, description=None):
    """Create a new security group

        https://docs.openstack.org/api-ref/compute/#create-security-group

        :param string name:
            Security group name
        :param integer description:
            Security group description
        """
    url = '/os-security-groups'
    params = {'name': name, 'description': description}
    return self.create(url, json={'security_group': params})['security_group']
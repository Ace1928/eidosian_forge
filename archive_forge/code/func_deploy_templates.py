from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import allocation as _allocation
from openstack.baremetal.v1 import chassis as _chassis
from openstack.baremetal.v1 import conductor as _conductor
from openstack.baremetal.v1 import deploy_templates as _deploytemplates
from openstack.baremetal.v1 import driver as _driver
from openstack.baremetal.v1 import node as _node
from openstack.baremetal.v1 import port as _port
from openstack.baremetal.v1 import port_group as _portgroup
from openstack.baremetal.v1 import volume_connector as _volumeconnector
from openstack.baremetal.v1 import volume_target as _volumetarget
from openstack import exceptions
from openstack import proxy
from openstack import utils
def deploy_templates(self, details=False, **query):
    """Retrieve a generator of deploy_templates.

        :param details: A boolean indicating whether the detailed information
            for every deploy_templates should be returned.
        :param dict query: Optional query parameters to be sent to
            restrict the deploy_templates to be returned.

        :returns: A generator of Deploy templates instances.
        """
    if details:
        query['detail'] = True
    return _deploytemplates.DeployTemplate.list(self, **query)
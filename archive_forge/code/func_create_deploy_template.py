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
def create_deploy_template(self, **attrs):
    """Create a new deploy_template from attributes.

        :param dict attrs: Keyword arguments that will be used to create a
            :class:`~openstack.baremetal.v1.deploy_templates.DeployTemplate`.

        :returns: The results of deploy_template creation.
        :rtype:
            :class:`~openstack.baremetal.v1.deploy_templates.DeployTemplate`.
        """
    return self._create(_deploytemplates.DeployTemplate, **attrs)
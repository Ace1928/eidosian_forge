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
def patch_deploy_template(self, deploy_template, patch):
    """Apply a JSON patch to the deploy_templates.

        :param deploy_templates: The value can be the ID of a
            deploy_template or a
            :class:`~openstack.baremetal.v1.deploy_templates.DeployTemplate`
            instance.

        :param patch: JSON patch to apply.

        :returns: The updated deploy_template.
        :rtype:
            :class:`~openstack.baremetal.v1.deploy_templates.DeployTemplate`
        """
    return self._get_resource(_deploytemplates.DeployTemplate, deploy_template).patch(self, patch)
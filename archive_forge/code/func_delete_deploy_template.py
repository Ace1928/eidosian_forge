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
def delete_deploy_template(self, deploy_template, ignore_missing=True):
    """Delete a deploy_template.

        :param deploy_template:The value can be
            either the ID of a deploy_template or a
            :class:`~openstack.baremetal.v1.deploy_templates.DeployTemplate`
            instance.

        :param bool ignore_missing: When set to ``False``,
            an exception:class:`~openstack.exceptions.ResourceNotFound`
            will be raised when the deploy_template
            could not be found.
            When set to ``True``, no
            exception will be raised when attempting
            to delete a non-existent
            deploy_template.

        :returns: The instance of the deploy_template which was deleted.
        :rtype:
            :class:`~openstack.baremetal.v1.deploy_templates.DeployTemplate`.
        """
    return self._delete(_deploytemplates.DeployTemplate, deploy_template, ignore_missing=ignore_missing)
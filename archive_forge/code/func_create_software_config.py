from openstack import exceptions
from openstack.orchestration.util import template_utils
from openstack.orchestration.v1 import resource as _resource
from openstack.orchestration.v1 import software_config as _sc
from openstack.orchestration.v1 import software_deployment as _sd
from openstack.orchestration.v1 import stack as _stack
from openstack.orchestration.v1 import stack_environment as _stack_environment
from openstack.orchestration.v1 import stack_event as _stack_event
from openstack.orchestration.v1 import stack_files as _stack_files
from openstack.orchestration.v1 import stack_template as _stack_template
from openstack.orchestration.v1 import template as _template
from openstack import proxy
from openstack import resource
def create_software_config(self, **attrs):
    """Create a new software config from attributes

        :param dict attrs: Keyword arguments which will be used to create a
            :class:`~openstack.orchestration.v1.software_config.SoftwareConfig`,
            comprised of the properties on the SoftwareConfig class.

        :returns: The results of software config creation
        :rtype:
            :class:`~openstack.orchestration.v1.software_config.SoftwareConfig`
        """
    return self._create(_sc.SoftwareConfig, **attrs)
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
def delete_software_config(self, software_config, ignore_missing=True):
    """Delete a software config

        :param software_config: The value can be either the ID of a software
            config or an instance of
            :class:`~openstack.orchestration.v1.software_config.SoftwareConfig`
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the software config does not exist.
            When set to ``True``, no exception will be set when
            attempting to delete a nonexistent software config.
        :returns: ``None``
        """
    self._delete(_sc.SoftwareConfig, software_config, ignore_missing=ignore_missing)
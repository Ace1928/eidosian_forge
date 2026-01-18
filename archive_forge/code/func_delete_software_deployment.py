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
def delete_software_deployment(self, software_deployment, ignore_missing=True):
    """Delete a software deployment

        :param software_deployment: The value can be either the ID of a
            software deployment or an instance of
            :class:`~openstack.orchestration.v1.software_deployment.SoftwareDeployment`
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the software deployment does not exist.
            When set to ``True``, no exception will be set when
            attempting to delete a nonexistent software deployment.
        :returns: ``None``
        """
    self._delete(_sd.SoftwareDeployment, software_deployment, ignore_missing=ignore_missing)
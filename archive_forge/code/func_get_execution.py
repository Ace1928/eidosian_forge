from openstack import proxy
from openstack.workflow.v2 import cron_trigger as _cron_trigger
from openstack.workflow.v2 import execution as _execution
from openstack.workflow.v2 import workflow as _workflow
def get_execution(self, *attrs):
    """Get a execution

        :param workflow_name: The name of target workflow to execute.
        :param execution: The value can be either the ID of a execution or a
            :class:`~openstack.workflow.v2.execution.Execution` instance.

        :returns: One :class:`~openstack.workflow.v2.execution.Execution`
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            execution matching the criteria could be found.
        """
    return self._get(_execution.Execution, *attrs)
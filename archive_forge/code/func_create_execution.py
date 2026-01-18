from openstack import proxy
from openstack.workflow.v2 import cron_trigger as _cron_trigger
from openstack.workflow.v2 import execution as _execution
from openstack.workflow.v2 import workflow as _workflow
def create_execution(self, **attrs):
    """Create a new execution from attributes

        :param workflow_name: The name of target workflow to execute.
        :param dict attrs: Keyword arguments which will be used to create
            a :class:`~openstack.workflow.v2.execution.Execution`,
            comprised of the properties on the Execution class.

        :returns: The results of execution creation
        :rtype: :class:`~openstack.workflow.v2.execution.Execution`
        """
    return self._create(_execution.Execution, **attrs)
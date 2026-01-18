from openstack import proxy
from openstack.workflow.v2 import cron_trigger as _cron_trigger
from openstack.workflow.v2 import execution as _execution
from openstack.workflow.v2 import workflow as _workflow
def delete_execution(self, value, ignore_missing=True):
    """Delete an execution

        :param value: The value can be either the name of a execution or a
            :class:`~openstack.workflow.v2.execute.Execution`
            instance.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the execution does not exist.
            When set to ``True``, no exception will be set when
            attempting to delete a nonexistent execution.

        :returns: ``None``
        """
    return self._delete(_execution.Execution, value, ignore_missing=ignore_missing)
from openstack import proxy
from openstack.workflow.v2 import cron_trigger as _cron_trigger
from openstack.workflow.v2 import execution as _execution
from openstack.workflow.v2 import workflow as _workflow
def delete_cron_trigger(self, value, ignore_missing=True):
    """Delete a cron trigger

        :param value: The value can be either the name of a cron trigger or a
            :class:`~openstack.workflow.v2.cron_trigger.CronTrigger`
            instance.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the cron trigger does not exist.
            When set to ``True``, no exception will be set when
            attempting to delete a nonexistent cron trigger.

        :returns: ``None``
        """
    return self._delete(_cron_trigger.CronTrigger, value, ignore_missing=ignore_missing)
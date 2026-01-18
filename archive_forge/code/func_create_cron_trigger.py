from openstack import proxy
from openstack.workflow.v2 import cron_trigger as _cron_trigger
from openstack.workflow.v2 import execution as _execution
from openstack.workflow.v2 import workflow as _workflow
def create_cron_trigger(self, **attrs):
    """Create a new cron trigger from attributes

        :param dict attrs: Keyword arguments which will be used to create
            a :class:`~openstack.workflow.v2.cron_trigger.CronTrigger`,
            comprised of the properties on the CronTrigger class.

        :returns: The results of cron trigger creation
        :rtype: :class:`~openstack.workflow.v2.cron_trigger.CronTrigger`
        """
    return self._create(_cron_trigger.CronTrigger, **attrs)
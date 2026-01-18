from openstack import proxy
from openstack.workflow.v2 import cron_trigger as _cron_trigger
from openstack.workflow.v2 import execution as _execution
from openstack.workflow.v2 import workflow as _workflow
def cron_triggers(self, *, all_projects=False, **query):
    """Retrieve a generator of cron triggers

        :param bool all_projects: When set to ``True``, list cron triggers from
            all projects. Admin-only by default.
        :param kwargs query: Optional query parameters to be sent to
            restrict the cron triggers to be returned. Available parameters
            include:

            * limit: Requests at most the specified number of items be
              returned from the query.
            * marker: Specifies the ID of the last-seen cron trigger. Use the
              limit parameter to make an initial limited request and use
              the ID of the last-seen cron trigger from the response as the
              marker parameter value in a subsequent limited request.

        :returns: A generator of CronTrigger instances.
        """
    if all_projects:
        query['all_projects'] = True
    return self._list(_cron_trigger.CronTrigger, **query)
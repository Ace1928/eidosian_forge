import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def count_pending_activity_tasks(self, domain, task_list):
    """
        Returns the estimated number of activity tasks in the
        specified task list. The count returned is an approximation
        and is not guaranteed to be exact. If you specify a task list
        that no activity task was ever scheduled in then 0 will be
        returned.

        :type domain: string
        :param domain: The name of the domain that contains the task list.

        :type task_list: string
        :param task_list: The name of the task list.

        :raises: UnknownResourceFault, SWFOperationNotPermittedError
        """
    return self.json_request('CountPendingActivityTasks', {'domain': domain, 'taskList': {'name': task_list}})
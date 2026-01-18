import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def respond_activity_task_completed(self, task_token, result=None):
    """
        Used by workers to tell the service that the ActivityTask
        identified by the taskToken completed successfully with a
        result (if provided).

        :type task_token: string
        :param task_token: The taskToken of the ActivityTask.

        :type result: string
        :param result: The result of the activity task. It is a free
            form string that is implementation specific.

        :raises: UnknownResourceFault, SWFOperationNotPermittedError
        """
    return self.json_request('RespondActivityTaskCompleted', {'taskToken': task_token, 'result': result})
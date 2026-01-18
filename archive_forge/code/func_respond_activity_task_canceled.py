import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def respond_activity_task_canceled(self, task_token, details=None):
    """
        Used by workers to tell the service that the ActivityTask
        identified by the taskToken was successfully
        canceled. Additional details can be optionally provided using
        the details argument.

        :type task_token: string
        :param task_token: The taskToken of the ActivityTask.

        :type details: string
        :param details: Optional detailed information about the failure.

        :raises: UnknownResourceFault, SWFOperationNotPermittedError
        """
    return self.json_request('RespondActivityTaskCanceled', {'taskToken': task_token, 'details': details})
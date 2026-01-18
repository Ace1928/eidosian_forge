import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def respond_activity_task_failed(self, task_token, details=None, reason=None):
    """
        Used by workers to tell the service that the ActivityTask
        identified by the taskToken has failed with reason (if
        specified).

        :type task_token: string
        :param task_token: The taskToken of the ActivityTask.

        :type details: string
        :param details: Optional detailed information about the failure.

        :type reason: string
        :param reason: Description of the error that may assist in diagnostics.

        :raises: UnknownResourceFault, SWFOperationNotPermittedError
        """
    return self.json_request('RespondActivityTaskFailed', {'taskToken': task_token, 'details': details, 'reason': reason})
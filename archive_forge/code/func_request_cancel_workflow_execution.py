import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def request_cancel_workflow_execution(self, domain, workflow_id, run_id=None):
    """
        Records a WorkflowExecutionCancelRequested event in the
        currently running workflow execution identified by the given
        domain, workflowId, and runId. This logically requests the
        cancellation of the workflow execution as a whole. It is up to
        the decider to take appropriate actions when it receives an
        execution history with this event.

        :type domain: string
        :param domain: The name of the domain containing the workflow
            execution to cancel.

        :type run_id: string
        :param run_id: The runId of the workflow execution to cancel.

        :type workflow_id: string
        :param workflow_id: The workflowId of the workflow execution
            to cancel.

        :raises: UnknownResourceFault, SWFOperationNotPermittedError
        """
    return self.json_request('RequestCancelWorkflowExecution', {'domain': domain, 'workflowId': workflow_id, 'runId': run_id})
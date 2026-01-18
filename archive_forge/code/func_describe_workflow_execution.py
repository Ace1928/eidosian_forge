import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def describe_workflow_execution(self, domain, run_id, workflow_id):
    """
        Returns information about the specified workflow execution
        including its type and some statistics.

        :type domain: string
        :param domain: The name of the domain containing the
            workflow execution.

        :type run_id: string
        :param run_id: A system generated unique identifier for the
            workflow execution.

        :type workflow_id: string
        :param workflow_id: The user defined identifier associated
            with the workflow execution.

        :raises: UnknownResourceFault, SWFOperationNotPermittedError
        """
    return self.json_request('DescribeWorkflowExecution', {'domain': domain, 'execution': {'runId': run_id, 'workflowId': workflow_id}})
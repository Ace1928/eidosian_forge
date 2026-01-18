import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def describe_workflow_type(self, domain, workflow_name, workflow_version):
    """
        Returns information about the specified workflow type. This
        includes configuration settings specified when the type was
        registered and other information such as creation date,
        current status, etc.

        :type domain: string
        :param domain: The name of the domain in which this workflow
            type is registered.

        :type workflow_name: string
        :param workflow_name: The name of the workflow type.

        :type workflow_version: string
        :param workflow_version: The version of the workflow type.

        :raises: UnknownResourceFault, SWFOperationNotPermittedError
        """
    return self.json_request('DescribeWorkflowType', {'domain': domain, 'workflowType': {'name': workflow_name, 'version': workflow_version}})
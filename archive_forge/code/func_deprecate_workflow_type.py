import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def deprecate_workflow_type(self, domain, workflow_name, workflow_version):
    """
        Deprecates the specified workflow type. After a workflow type
        has been deprecated, you cannot create new executions of that
        type. Executions that were started before the type was
        deprecated will continue to run. A deprecated workflow type
        may still be used when calling visibility actions.

        :type domain: string
        :param domain: The name of the domain in which the workflow
            type is registered.

        :type workflow_name: string
        :param workflow_name: The name of the workflow type.

        :type workflow_version: string
        :param workflow_version: The version of the workflow type.

        :raises: UnknownResourceFault, TypeDeprecatedFault,
            SWFOperationNotPermittedError
        """
    return self.json_request('DeprecateWorkflowType', {'domain': domain, 'workflowType': {'name': workflow_name, 'version': workflow_version}})
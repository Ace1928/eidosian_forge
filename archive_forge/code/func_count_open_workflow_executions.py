import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def count_open_workflow_executions(self, domain, latest_date, oldest_date, tag=None, workflow_id=None, workflow_name=None, workflow_version=None):
    """
        Returns the number of open workflow executions within the
        given domain that meet the specified filtering criteria.

        .. note:
            workflow_id, workflow_name/workflow_version and tag are mutually
            exclusive. You can specify at most one of these in a request.

        :type domain: string
        :param domain: The name of the domain containing the
            workflow executions to count.

        :type latest_date: timestamp
        :param latest_date: Specifies the latest start or close date
            and time to return.

        :type oldest_date: timestamp
        :param oldest_date: Specifies the oldest start or close date
            and time to return.

        :type workflow_name: string
        :param workflow_name: Name of the workflow type to filter on.

        :type workflow_version: string
        :param workflow_version: Version of the workflow type to filter on.

        :type tag: string
        :param tag: If specified, only executions that have a tag
            that matches the filter are counted.

        :type workflow_id: string
        :param workflow_id: If specified, only workflow executions
            matching the workflow_id are counted.

        :raises: UnknownResourceFault, SWFOperationNotPermittedError
        """
    return self.json_request('CountOpenWorkflowExecutions', {'domain': domain, 'startTimeFilter': {'oldestDate': oldest_date, 'latestDate': latest_date}, 'typeFilter': {'name': workflow_name, 'version': workflow_version}, 'executionFilter': {'workflowId': workflow_id}, 'tagFilter': {'tag': tag}})
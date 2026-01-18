import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def count_closed_workflow_executions(self, domain, start_latest_date=None, start_oldest_date=None, close_latest_date=None, close_oldest_date=None, close_status=None, tag=None, workflow_id=None, workflow_name=None, workflow_version=None):
    """
        Returns the number of closed workflow executions within the
        given domain that meet the specified filtering criteria.

        .. note:
            close_status, workflow_id, workflow_name/workflow_version
            and tag are mutually exclusive. You can specify at most
            one of these in a request.

        .. note:
            start_latest_date/start_oldest_date and
            close_latest_date/close_oldest_date are mutually
            exclusive. You can specify at most one of these in a request.

        :type domain: string
        :param domain: The name of the domain containing the
            workflow executions to count.

        :type start_latest_date: timestamp
        :param start_latest_date: If specified, only workflow executions
            that meet the start time criteria of the filter are counted.

        :type start_oldest_date: timestamp
        :param start_oldest_date: If specified, only workflow executions
            that meet the start time criteria of the filter are counted.

        :type close_latest_date: timestamp
        :param close_latest_date: If specified, only workflow executions
            that meet the close time criteria of the filter are counted.

        :type close_oldest_date: timestamp
        :param close_oldest_date: If specified, only workflow executions
            that meet the close time criteria of the filter are counted.

        :type close_status: string
        :param close_status: The close status that must match the close status
            of an execution for it to meet the criteria of this filter.
            Valid values are:

            * COMPLETED
            * FAILED
            * CANCELED
            * TERMINATED
            * CONTINUED_AS_NEW
            * TIMED_OUT

        :type tag: string
        :param tag: If specified, only executions that have a tag
            that matches the filter are counted.

        :type workflow_id: string
        :param workflow_id: If specified, only workflow executions
            matching the workflow_id are counted.

        :type workflow_name: string
        :param workflow_name: Name of the workflow type to filter on.

        :type workflow_version: string
        :param workflow_version: Version of the workflow type to filter on.

        :raises: UnknownResourceFault, SWFOperationNotPermittedError
        """
    return self.json_request('CountClosedWorkflowExecutions', {'domain': domain, 'startTimeFilter': {'oldestDate': start_oldest_date, 'latestDate': start_latest_date}, 'closeTimeFilter': {'oldestDate': close_oldest_date, 'latestDate': close_latest_date}, 'closeStatusFilter': {'status': close_status}, 'tagFilter': {'tag': tag}, 'typeFilter': {'name': workflow_name, 'version': workflow_version}, 'executionFilter': {'workflowId': workflow_id}})
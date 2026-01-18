import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def list_open_workflow_executions(self, domain, oldest_date, latest_date=None, tag=None, workflow_id=None, workflow_name=None, workflow_version=None, maximum_page_size=None, next_page_token=None, reverse_order=None):
    """
        Returns the list of open workflow executions within the
        given domain that meet the specified filtering criteria.

        .. note:
            workflow_id, workflow_name/workflow_version
            and tag are mutually exclusive. You can specify at most
            one of these in a request.

        :type domain: string
        :param domain: The name of the domain containing the
            workflow executions to count.

        :type latest_date: timestamp
        :param latest_date: Specifies the latest start or close date
            and time to return.

        :type oldest_date: timestamp
        :param oldest_date: Specifies the oldest start or close date
            and time to return.

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

        :type maximum_page_size: integer
        :param maximum_page_size: The maximum number of results
            returned in each page. The default is 100, but the caller can
            override this value to a page size smaller than the
            default. You cannot specify a page size greater than 100.

        :type next_page_token: string
        :param next_page_token: If on a previous call to this method a
            NextPageToken was returned, the results are being
            paginated. To get the next page of results, repeat the call
            with the returned token and all other arguments unchanged.

        :type reverse_order: boolean
        :param reverse_order: When set to true, returns the results in
            reverse order. By default the results are returned in
            descending order of the start or the close time of the
            executions.

        :raises: UnknownResourceFault, SWFOperationNotPermittedError

        """
    return self.json_request('ListOpenWorkflowExecutions', {'domain': domain, 'startTimeFilter': {'oldestDate': oldest_date, 'latestDate': latest_date}, 'tagFilter': {'tag': tag}, 'typeFilter': {'name': workflow_name, 'version': workflow_version}, 'executionFilter': {'workflowId': workflow_id}, 'maximumPageSize': maximum_page_size, 'nextPageToken': next_page_token, 'reverseOrder': reverse_order})
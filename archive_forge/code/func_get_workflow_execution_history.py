import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def get_workflow_execution_history(self, domain, run_id, workflow_id, maximum_page_size=None, next_page_token=None, reverse_order=None):
    """
        Returns the history of the specified workflow execution. The
        results may be split into multiple pages. To retrieve
        subsequent pages, make the call again using the nextPageToken
        returned by the initial call.

        :type domain: string
        :param domain: The name of the domain containing the
            workflow execution.

        :type run_id: string
        :param run_id: A system generated unique identifier for the
            workflow execution.

        :type workflow_id: string
        :param workflow_id: The user defined identifier associated
            with the workflow execution.

        :type maximum_page_size: integer
        :param maximum_page_size: Specifies the maximum number of
            history events returned in one page. The next page in the
            result is identified by the NextPageToken returned. By default
            100 history events are returned in a page but the caller can
            override this value to a page size smaller than the
            default. You cannot specify a page size larger than 100.

        :type next_page_token: string
        :param next_page_token: If a NextPageToken is returned, the
            result has more than one pages. To get the next page, repeat
            the call and specify the nextPageToken with all other
            arguments unchanged.

        :type reverse_order: boolean
        :param reverse_order: When set to true, returns the events in
            reverse order. By default the results are returned in
            ascending order of the eventTimeStamp of the events.

        :raises: UnknownResourceFault, SWFOperationNotPermittedError
        """
    return self.json_request('GetWorkflowExecutionHistory', {'domain': domain, 'execution': {'runId': run_id, 'workflowId': workflow_id}, 'maximumPageSize': maximum_page_size, 'nextPageToken': next_page_token, 'reverseOrder': reverse_order})
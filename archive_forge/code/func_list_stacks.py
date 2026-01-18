import boto
from boto.cloudformation.stack import Stack, StackSummary, StackEvent
from boto.cloudformation.stack import StackResource, StackResourceSummary
from boto.cloudformation.template import Template
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
def list_stacks(self, stack_status_filters=None, next_token=None):
    """
        Returns the summary information for stacks whose status
        matches the specified StackStatusFilter. Summary information
        for stacks that have been deleted is kept for 90 days after
        the stack is deleted. If no StackStatusFilter is specified,
        summary information for all stacks is returned (including
        existing stacks and stacks that have been deleted).

        :type next_token: string
        :param next_token: String that identifies the start of the next list of
            stacks, if there is one.
        Default: There is no default value.

        :type stack_status_filter: list
        :param stack_status_filter: Stack status to use as a filter. Specify
            one or more stack status codes to list only stacks with the
            specified status codes. For a complete list of stack status codes,
            see the `StackStatus` parameter of the Stack data type.

        """
    params = {}
    if next_token:
        params['NextToken'] = next_token
    if stack_status_filters and len(stack_status_filters) > 0:
        self.build_list_params(params, stack_status_filters, 'StackStatusFilter.member')
    return self.get_list('ListStacks', params, [('member', StackSummary)])
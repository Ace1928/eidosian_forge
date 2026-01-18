import boto
from boto.cloudformation.stack import Stack, StackSummary, StackEvent
from boto.cloudformation.stack import StackResource, StackResourceSummary
from boto.cloudformation.template import Template
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
def describe_stack_events(self, stack_name_or_id=None, next_token=None):
    """
        Returns all stack related events for a specified stack. For
        more information about a stack's event history, go to
        `Stacks`_ in the AWS CloudFormation User Guide.
        Events are returned, even if the stack never existed or has
        been successfully deleted.

        :type stack_name_or_id: string
        :param stack_name_or_id: The name or the unique identifier associated
            with the stack.
        Default: There is no default value.

        :type next_token: string
        :param next_token: String that identifies the start of the next list of
            events, if there is one.
        Default: There is no default value.

        """
    params = {}
    if stack_name_or_id:
        params['StackName'] = stack_name_or_id
    if next_token:
        params['NextToken'] = next_token
    return self.get_list('DescribeStackEvents', params, [('member', StackEvent)])
import boto
from boto.cloudformation.stack import Stack, StackSummary, StackEvent
from boto.cloudformation.stack import StackResource, StackResourceSummary
from boto.cloudformation.template import Template
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
def describe_stacks(self, stack_name_or_id=None, next_token=None):
    """
        Returns the description for the specified stack; if no stack
        name was specified, then it returns the description for all
        the stacks created.

        :type stack_name_or_id: string
        :param stack_name_or_id: The name or the unique identifier associated
            with the stack.
        Default: There is no default value.

        :type next_token: string
        :param next_token: String that identifies the start of the next list of
            stacks, if there is one.

        """
    params = {}
    if stack_name_or_id:
        params['StackName'] = stack_name_or_id
    if next_token is not None:
        params['NextToken'] = next_token
    return self.get_list('DescribeStacks', params, [('member', Stack)])
import boto
from boto.cloudformation.stack import Stack, StackSummary, StackEvent
from boto.cloudformation.stack import StackResource, StackResourceSummary
from boto.cloudformation.template import Template
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
def list_stack_resources(self, stack_name_or_id, next_token=None):
    """
        Returns descriptions of all resources of the specified stack.

        For deleted stacks, ListStackResources returns resource
        information for up to 90 days after the stack has been
        deleted.

        :type stack_name_or_id: string
        :param stack_name_or_id: The name or the unique identifier associated
            with the stack, which are not always interchangeable:

        + Running stacks: You can specify either the stack's name or its unique
              stack ID.
        + Deleted stacks: You must specify the unique stack ID.


        Default: There is no default value.

        :type next_token: string
        :param next_token: String that identifies the start of the next list of
            stack resource summaries, if there is one.
        Default: There is no default value.

        """
    params = {'StackName': stack_name_or_id}
    if next_token:
        params['NextToken'] = next_token
    return self.get_list('ListStackResources', params, [('member', StackResourceSummary)])
import boto
from boto.cloudformation.stack import Stack, StackSummary, StackEvent
from boto.cloudformation.stack import StackResource, StackResourceSummary
from boto.cloudformation.template import Template
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
def describe_stack_resource(self, stack_name_or_id, logical_resource_id):
    """
        Returns a description of the specified resource in the
        specified stack.

        For deleted stacks, DescribeStackResource returns resource
        information for up to 90 days after the stack has been
        deleted.

        :type stack_name_or_id: string
        :param stack_name_or_id: The name or the unique identifier associated
            with the stack.
        Default: There is no default value.

        :type logical_resource_id: string
        :param logical_resource_id: The logical name of the resource as
            specified in the template.
        Default: There is no default value.

        """
    params = {'ContentType': 'JSON', 'StackName': stack_name_or_id, 'LogicalResourceId': logical_resource_id}
    return self._do_request('DescribeStackResource', params, '/', 'GET')
import boto
from boto.cloudformation.stack import Stack, StackSummary, StackEvent
from boto.cloudformation.stack import StackResource, StackResourceSummary
from boto.cloudformation.template import Template
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
def delete_stack(self, stack_name_or_id):
    """
        Deletes a specified stack. Once the call completes
        successfully, stack deletion starts. Deleted stacks do not
        show up in the DescribeStacks API if the deletion has been
        completed successfully.

        :type stack_name_or_id: string
        :param stack_name_or_id: The name or the unique identifier associated
            with the stack.

        """
    params = {'ContentType': 'JSON', 'StackName': stack_name_or_id}
    return self._do_request('DeleteStack', params, '/', 'GET')
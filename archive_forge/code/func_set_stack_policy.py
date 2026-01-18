import boto
from boto.cloudformation.stack import Stack, StackSummary, StackEvent
from boto.cloudformation.stack import StackResource, StackResourceSummary
from boto.cloudformation.template import Template
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
def set_stack_policy(self, stack_name_or_id, stack_policy_body=None, stack_policy_url=None):
    """
        Sets a stack policy for a specified stack.

        :type stack_name_or_id: string
        :param stack_name_or_id: The name or stack ID that you want to
            associate a policy with.

        :type stack_policy_body: string
        :param stack_policy_body: Structure containing the stack policy body.
            (For more information, go to ` Prevent Updates to Stack Resources`_
            in the AWS CloudFormation User Guide.)
        You must pass `StackPolicyBody` or `StackPolicyURL`. If both are
            passed, only `StackPolicyBody` is used.

        :type stack_policy_url: string
        :param stack_policy_url: Location of a file containing the stack
            policy. The URL must point to a policy (max size: 16KB) located in
            an S3 bucket in the same region as the stack. You must pass
            `StackPolicyBody` or `StackPolicyURL`. If both are passed, only
            `StackPolicyBody` is used.

        """
    params = {'ContentType': 'JSON', 'StackName': stack_name_or_id}
    if stack_policy_body is not None:
        params['StackPolicyBody'] = stack_policy_body
    if stack_policy_url is not None:
        params['StackPolicyURL'] = stack_policy_url
    response = self._do_request('SetStackPolicy', params, '/', 'POST')
    return response['SetStackPolicyResponse']
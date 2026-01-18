import boto
from boto.cloudformation.stack import Stack, StackSummary, StackEvent
from boto.cloudformation.stack import StackResource, StackResourceSummary
from boto.cloudformation.template import Template
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
def create_stack(self, stack_name, template_body=None, template_url=None, parameters=None, notification_arns=None, disable_rollback=None, timeout_in_minutes=None, capabilities=None, tags=None, on_failure=None, stack_policy_body=None, stack_policy_url=None):
    """
        Creates a stack as specified in the template. After the call
        completes successfully, the stack creation starts. You can
        check the status of the stack via the DescribeStacks API.
        Currently, the limit for stacks is 20 stacks per account per
        region.

        :type stack_name: string
        :param stack_name:
        The name associated with the stack. The name must be unique within your
            AWS account.

        Must contain only alphanumeric characters (case sensitive) and start
            with an alpha character. Maximum length of the name is 255
            characters.

        :type template_body: string
        :param template_body: Structure containing the template body. (For more
            information, go to `Template Anatomy`_ in the AWS CloudFormation
            User Guide.)
        Conditional: You must pass `TemplateBody` or `TemplateURL`. If both are
            passed, only `TemplateBody` is used.

        :type template_url: string
        :param template_url: Location of file containing the template body. The
            URL must point to a template (max size: 307,200 bytes) located in
            an S3 bucket in the same region as the stack. For more information,
            go to the `Template Anatomy`_ in the AWS CloudFormation User Guide.
        Conditional: You must pass `TemplateURL` or `TemplateBody`. If both are
            passed, only `TemplateBody` is used.

        :type parameters: list
        :param parameters: A list of key/value tuples that specify input
            parameters for the stack.

        :type disable_rollback: boolean
        :param disable_rollback: Set to `True` to disable rollback of the stack
            if stack creation failed. You can specify either `DisableRollback`
            or `OnFailure`, but not both.
        Default: `False`

        :type timeout_in_minutes: integer
        :param timeout_in_minutes: The amount of time that can pass before the
            stack status becomes CREATE_FAILED; if `DisableRollback` is not set
            or is set to `False`, the stack will be rolled back.

        :type notification_arns: list
        :param notification_arns: The Simple Notification Service (SNS) topic
            ARNs to publish stack related events. You can find your SNS topic
            ARNs using the `SNS console`_ or your Command Line Interface (CLI).

        :type capabilities: list
        :param capabilities: The list of capabilities that you want to allow in
            the stack. If your template contains certain resources, you must
            specify the CAPABILITY_IAM value for this parameter; otherwise,
            this action returns an InsufficientCapabilities error. The
            following resources require you to specify the capabilities
            parameter: `AWS::CloudFormation::Stack`_, `AWS::IAM::AccessKey`_,
            `AWS::IAM::Group`_, `AWS::IAM::InstanceProfile`_,
            `AWS::IAM::Policy`_, `AWS::IAM::Role`_, `AWS::IAM::User`_, and
            `AWS::IAM::UserToGroupAddition`_.

        :type on_failure: string
        :param on_failure: Determines what action will be taken if stack
            creation fails. This must be one of: DO_NOTHING, ROLLBACK, or
            DELETE. You can specify either `OnFailure` or `DisableRollback`,
            but not both.
        Default: `ROLLBACK`

        :type stack_policy_body: string
        :param stack_policy_body: Structure containing the stack policy body.
            (For more information, go to ` Prevent Updates to Stack Resources`_
            in the AWS CloudFormation User Guide.)
        If you pass `StackPolicyBody` and `StackPolicyURL`, only
            `StackPolicyBody` is used.

        :type stack_policy_url: string
        :param stack_policy_url: Location of a file containing the stack
            policy. The URL must point to a policy (max size: 16KB) located in
            an S3 bucket in the same region as the stack. If you pass
            `StackPolicyBody` and `StackPolicyURL`, only `StackPolicyBody` is
            used.

        :type tags: dict
        :param tags: A set of user-defined `Tags` to associate with this stack,
            represented by key/value pairs. Tags defined for the stack are
            propagated to EC2 resources that are created as part of the stack.
            A maximum number of 10 tags can be specified.
        """
    params = self._build_create_or_update_params(stack_name, template_body, template_url, parameters, disable_rollback, timeout_in_minutes, notification_arns, capabilities, on_failure, stack_policy_body, stack_policy_url, tags)
    body = self._do_request('CreateStack', params, '/', 'POST')
    return body['CreateStackResponse']['CreateStackResult']['StackId']
import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def start_workflow_execution(self, domain, workflow_id, workflow_name, workflow_version, task_list=None, child_policy=None, execution_start_to_close_timeout=None, input=None, tag_list=None, task_start_to_close_timeout=None):
    """
        Starts an execution of the workflow type in the specified
        domain using the provided workflowId and input data.

        :type domain: string
        :param domain: The name of the domain in which the workflow
            execution is created.

        :type workflow_id: string
        :param workflow_id: The user defined identifier associated with
            the workflow execution. You can use this to associate a
            custom identifier with the workflow execution. You may
            specify the same identifier if a workflow execution is
            logically a restart of a previous execution. You cannot
            have two open workflow executions with the same workflowId
            at the same time.

        :type workflow_name: string
        :param workflow_name: The name of the workflow type.

        :type workflow_version: string
        :param workflow_version: The version of the workflow type.

        :type task_list: string
        :param task_list: The task list to use for the decision tasks
            generated for this workflow execution. This overrides the
            defaultTaskList specified when registering the workflow type.

        :type child_policy: string
        :param child_policy: If set, specifies the policy to use for the
            child workflow executions of this workflow execution if it
            is terminated, by calling the TerminateWorkflowExecution
            action explicitly or due to an expired timeout. This policy
            overrides the default child policy specified when registering
            the workflow type using RegisterWorkflowType. The supported
            child policies are:

             * TERMINATE: the child executions will be terminated.
             * REQUEST_CANCEL: a request to cancel will be attempted
                 for each child execution by recording a
                 WorkflowExecutionCancelRequested event in its history.
                 It is up to the decider to take appropriate actions
                 when it receives an execution history with this event.
             * ABANDON: no action will be taken. The child executions
                 will continue to run.

        :type execution_start_to_close_timeout: string
        :param execution_start_to_close_timeout: The total duration for
            this workflow execution. This overrides the
            defaultExecutionStartToCloseTimeout specified when
            registering the workflow type.

        :type input: string
        :param input: The input for the workflow
            execution. This is a free form string which should be
            meaningful to the workflow you are starting. This input is
            made available to the new workflow execution in the
            WorkflowExecutionStarted history event.

        :type tag_list: list :param tag_list: The list of tags to
            associate with the workflow execution. You can specify a
            maximum of 5 tags. You can list workflow executions with a
            specific tag by calling list_open_workflow_executions or
            list_closed_workflow_executions and specifying a TagFilter.

        :type task_start_to_close_timeout: string :param
        task_start_to_close_timeout: Specifies the maximum duration of
            decision tasks for this workflow execution. This parameter
            overrides the defaultTaskStartToCloseTimout specified when
            registering the workflow type using register_workflow_type.

        :raises: UnknownResourceFault, TypeDeprecatedFault,
            SWFWorkflowExecutionAlreadyStartedError, SWFLimitExceededError,
            SWFOperationNotPermittedError, DefaultUndefinedFault
        """
    return self.json_request('StartWorkflowExecution', {'domain': domain, 'workflowId': workflow_id, 'workflowType': {'name': workflow_name, 'version': workflow_version}, 'taskList': {'name': task_list}, 'childPolicy': child_policy, 'executionStartToCloseTimeout': execution_start_to_close_timeout, 'input': input, 'tagList': tag_list, 'taskStartToCloseTimeout': task_start_to_close_timeout})
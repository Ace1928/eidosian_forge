import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.datapipeline import exceptions
def set_task_status(self, task_id, task_status, error_id=None, error_message=None, error_stack_trace=None):
    """
        Notifies AWS Data Pipeline that a task is completed and
        provides information about the final status. The task runner
        calls this action regardless of whether the task was
        sucessful. The task runner does not need to call SetTaskStatus
        for tasks that are canceled by the web service during a call
        to ReportTaskProgress.

        :type task_id: string
        :param task_id: Identifies the task assigned to the task runner. This
            value is set in the TaskObject that is returned by the PollForTask
            action.

        :type task_status: string
        :param task_status: If `FINISHED`, the task successfully completed. If
            `FAILED` the task ended unsuccessfully. The `FALSE` value is used
            by preconditions.

        :type error_id: string
        :param error_id: If an error occurred during the task, this value
            specifies an id value that represents the error. This value is set
            on the physical attempt object. It is used to display error
            information to the user. It should not start with string "Service_"
            which is reserved by the system.

        :type error_message: string
        :param error_message: If an error occurred during the task, this value
            specifies a text description of the error. This value is set on the
            physical attempt object. It is used to display error information to
            the user. The web service does not parse this value.

        :type error_stack_trace: string
        :param error_stack_trace: If an error occurred during the task, this
            value specifies the stack trace associated with the error. This
            value is set on the physical attempt object. It is used to display
            error information to the user. The web service does not parse this
            value.

        """
    params = {'taskId': task_id, 'taskStatus': task_status}
    if error_id is not None:
        params['errorId'] = error_id
    if error_message is not None:
        params['errorMessage'] = error_message
    if error_stack_trace is not None:
        params['errorStackTrace'] = error_stack_trace
    return self.make_request(action='SetTaskStatus', body=json.dumps(params))
import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.datapipeline import exceptions
def report_task_progress(self, task_id):
    """
        Updates the AWS Data Pipeline service on the progress of the
        calling task runner. When the task runner is assigned a task,
        it should call ReportTaskProgress to acknowledge that it has
        the task within 2 minutes. If the web service does not recieve
        this acknowledgement within the 2 minute window, it will
        assign the task in a subsequent PollForTask call. After this
        initial acknowledgement, the task runner only needs to report
        progress every 15 minutes to maintain its ownership of the
        task. You can change this reporting time from 15 minutes by
        specifying a `reportProgressTimeout` field in your pipeline.
        If a task runner does not report its status after 5 minutes,
        AWS Data Pipeline will assume that the task runner is unable
        to process the task and will reassign the task in a subsequent
        response to PollForTask. task runners should call
        ReportTaskProgress every 60 seconds.

        :type task_id: string
        :param task_id: Identifier of the task assigned to the task runner.
            This value is provided in the TaskObject that the service returns
            with the response for the PollForTask action.

        """
    params = {'taskId': task_id}
    return self.make_request(action='ReportTaskProgress', body=json.dumps(params))
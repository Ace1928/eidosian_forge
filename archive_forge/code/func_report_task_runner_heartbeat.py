import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.datapipeline import exceptions
def report_task_runner_heartbeat(self, taskrunner_id, worker_group=None, hostname=None):
    """
        Task runners call ReportTaskRunnerHeartbeat every 15 minutes
        to indicate that they are operational. In the case of AWS Data
        Pipeline Task Runner launched on a resource managed by AWS
        Data Pipeline, the web service can use this call to detect
        when the task runner application has failed and restart a new
        instance.

        :type taskrunner_id: string
        :param taskrunner_id: The identifier of the task runner. This value
            should be unique across your AWS account. In the case of AWS Data
            Pipeline Task Runner launched on a resource managed by AWS Data
            Pipeline, the web service provides a unique identifier when it
            launches the application. If you have written a custom task runner,
            you should assign a unique identifier for the task runner.

        :type worker_group: string
        :param worker_group: Indicates the type of task the task runner is
            configured to accept and process. The worker group is set as a
            field on objects in the pipeline when they are created. You can
            only specify a single value for `workerGroup` in the call to
            ReportTaskRunnerHeartbeat. There are no wildcard values permitted
            in `workerGroup`, the string must be an exact, case-sensitive,
            match.

        :type hostname: string
        :param hostname: The public DNS name of the calling task runner.

        """
    params = {'taskrunnerId': taskrunner_id}
    if worker_group is not None:
        params['workerGroup'] = worker_group
    if hostname is not None:
        params['hostname'] = hostname
    return self.make_request(action='ReportTaskRunnerHeartbeat', body=json.dumps(params))
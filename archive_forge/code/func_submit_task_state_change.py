import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.ec2containerservice import exceptions
def submit_task_state_change(self, cluster=None, task=None, status=None, reason=None):
    """
        This action is only used by the Amazon EC2 Container Service
        agent, and it is not intended for use outside of the agent.


        Sent to acknowledge that a task changed states.

        :type cluster: string
        :param cluster: The short name or full Amazon Resource Name (ARN) of
            the cluster that hosts the task.

        :type task: string
        :param task: The task UUID or full Amazon Resource Name (ARN) of the
            task in the state change request.

        :type status: string
        :param status: The status of the state change request.

        :type reason: string
        :param reason: The reason for the state change request.

        """
    params = {}
    if cluster is not None:
        params['cluster'] = cluster
    if task is not None:
        params['task'] = task
    if status is not None:
        params['status'] = status
    if reason is not None:
        params['reason'] = reason
    return self._make_request(action='SubmitTaskStateChange', verb='POST', path='/', params=params)
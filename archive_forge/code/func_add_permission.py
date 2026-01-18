import boto
from boto.connection import AWSQueryConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.queue import Queue
from boto.sqs.message import Message
from boto.sqs.attributes import Attributes
from boto.sqs.batchresults import BatchResults
from boto.exception import SQSError, BotoServerError
def add_permission(self, queue, label, aws_account_id, action_name):
    """
        Add a permission to a queue.

        :type queue: :class:`boto.sqs.queue.Queue`
        :param queue: The queue object

        :type label: str or unicode
        :param label: A unique identification of the permission you are setting.
            Maximum of 80 characters ``[0-9a-zA-Z_-]``
            Example, AliceSendMessage

        :type aws_account_id: str or unicode
        :param principal_id: The AWS account number of the principal
            who will be given permission.  The principal must have an
            AWS account, but does not need to be signed up for Amazon
            SQS. For information about locating the AWS account
            identification.

        :type action_name: str or unicode
        :param action_name: The action.  Valid choices are:
            * *
            * SendMessage
            * ReceiveMessage
            * DeleteMessage
            * ChangeMessageVisibility
            * GetQueueAttributes

        :rtype: bool
        :return: True if successful, False otherwise.

        """
    params = {'Label': label, 'AWSAccountId': aws_account_id, 'ActionName': action_name}
    return self.get_status('AddPermission', params, queue.id)
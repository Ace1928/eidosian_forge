import xml.sax
import datetime
import itertools
from boto import handler
from boto import config
from boto.mturk.price import Price
import boto.mturk.notification
from boto.connection import AWSQueryConnection
from boto.exception import EC2ResponseError
from boto.resultset import ResultSet
from boto.mturk.question import QuestionForm, ExternalQuestion, HTMLQuestion
def approve_rejected_assignment(self, assignment_id, feedback=None):
    """
        """
    params = {'AssignmentId': assignment_id}
    if feedback:
        params['RequesterFeedback'] = feedback
    return self._process_request('ApproveRejectedAssignment', params)
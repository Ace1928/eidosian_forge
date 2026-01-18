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
def notify_workers(self, worker_ids, subject, message_text):
    """
        Send a text message to workers.
        """
    params = {'Subject': subject, 'MessageText': message_text}
    self.build_list_params(params, worker_ids, 'WorkerId')
    return self._process_request('NotifyWorkers', params)
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
def grant_qualification(self, qualification_request_id, integer_value=1):
    """TODO: Document."""
    params = {'QualificationRequestId': qualification_request_id, 'IntegerValue': integer_value}
    return self._process_request('GrantQualification', params)
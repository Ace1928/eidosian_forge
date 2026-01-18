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
def register_hit_type(self, title, description, reward, duration, keywords=None, approval_delay=None, qual_req=None):
    """
        Register a new HIT Type
        title, description are strings
        reward is a Price object
        duration can be a timedelta, or an object castable to an int
        """
    params = dict(Title=title, Description=description, AssignmentDurationInSeconds=self.duration_as_seconds(duration))
    params.update(MTurkConnection.get_price_as_price(reward).get_as_params('Reward'))
    if keywords:
        params['Keywords'] = self.get_keywords_as_string(keywords)
    if approval_delay is not None:
        d = self.duration_as_seconds(approval_delay)
        params['AutoApprovalDelayInSeconds'] = d
    if qual_req is not None:
        params.update(qual_req.get_as_params())
    return self._process_request('RegisterHITType', params, [('HITTypeId', HITTypeId)])
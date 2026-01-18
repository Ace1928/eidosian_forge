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
def grant_bonus(self, worker_id, assignment_id, bonus_price, reason):
    """
        Issues a payment of money from your account to a Worker.  To
        be eligible for a bonus, the Worker must have submitted
        results for one of your HITs, and have had those results
        approved or rejected. This payment happens separately from the
        reward you pay to the Worker when you approve the Worker's
        assignment.  The Bonus must be passed in as an instance of the
        Price object.
        """
    params = bonus_price.get_as_params('BonusAmount', 1)
    params['WorkerId'] = worker_id
    params['AssignmentId'] = assignment_id
    params['Reason'] = reason
    return self._process_request('GrantBonus', params)
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
def extend_hit(self, hit_id, assignments_increment=None, expiration_increment=None):
    """
        Increase the maximum number of assignments, or extend the
        expiration date, of an existing HIT.

        NOTE: If a HIT has a status of Reviewable and the HIT is
        extended to make it Available, the HIT will not be returned by
        GetReviewableHITs, and its submitted assignments will not be
        returned by GetAssignmentsForHIT, until the HIT is Reviewable
        again.  Assignment auto-approval will still happen on its
        original schedule, even if the HIT has been extended. Be sure
        to retrieve and approve (or reject) submitted assignments
        before extending the HIT, if so desired.
        """
    if assignments_increment is None and expiration_increment is None or (assignments_increment is not None and expiration_increment is not None):
        raise ValueError('Must specify either assignments_increment or expiration_increment, but not both')
    params = {'HITId': hit_id}
    if assignments_increment:
        params['MaxAssignmentsIncrement'] = assignments_increment
    if expiration_increment:
        params['ExpirationIncrementInSeconds'] = expiration_increment
    return self._process_request('ExtendHIT', params)
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
def expire_hit(self, hit_id):
    """
        Expire a HIT that is no longer needed.

        The effect is identical to the HIT expiring on its own. The
        HIT no longer appears on the Mechanical Turk web site, and no
        new Workers are allowed to accept the HIT. Workers who have
        accepted the HIT prior to expiration are allowed to complete
        it or return it, or allow the assignment duration to elapse
        (abandon the HIT). Once all remaining assignments have been
        submitted, the expired HIT becomes"reviewable", and will be
        returned by a call to GetReviewableHITs.
        """
    params = {'HITId': hit_id}
    return self._process_request('ForceExpireHIT', params)
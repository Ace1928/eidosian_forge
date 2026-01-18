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
def dispose_hit(self, hit_id):
    """
        Dispose of a HIT that is no longer needed.

        Only HITs in the "reviewable" state, with all submitted
        assignments approved or rejected, can be disposed. A Requester
        can call GetReviewableHITs to determine which HITs are
        reviewable, then call GetAssignmentsForHIT to retrieve the
        assignments.  Disposing of a HIT removes the HIT from the
        results of a call to GetReviewableHITs.  """
    params = {'HITId': hit_id}
    return self._process_request('DisposeHIT', params)
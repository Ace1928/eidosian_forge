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
def set_reviewing(self, hit_id, revert=None):
    """
        Update a HIT with a status of Reviewable to have a status of Reviewing,
        or reverts a Reviewing HIT back to the Reviewable status.

        Only HITs with a status of Reviewable can be updated with a status of
        Reviewing.  Similarly, only Reviewing HITs can be reverted back to a
        status of Reviewable.
        """
    params = {'HITId': hit_id}
    if revert:
        params['Revert'] = revert
    return self._process_request('SetHITAsReviewing', params)
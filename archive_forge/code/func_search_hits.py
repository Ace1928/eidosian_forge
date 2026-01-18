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
def search_hits(self, sort_by='CreationTime', sort_direction='Ascending', page_size=10, page_number=1, response_groups=None):
    """
        Return a page of a Requester's HITs, on behalf of the Requester.
        The operation returns HITs of any status, except for HITs that
        have been disposed with the DisposeHIT operation.
        Note:
        The SearchHITs operation does not accept any search parameters
        that filter the results.
        """
    params = {'SortProperty': sort_by, 'SortDirection': sort_direction, 'PageSize': page_size, 'PageNumber': page_number}
    if response_groups:
        self.build_list_params(params, response_groups, 'ResponseGroup')
    return self._process_request('SearchHITs', params, [('HIT', HIT)])
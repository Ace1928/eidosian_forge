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
def get_assignment(self, assignment_id, response_groups=None):
    """
        Retrieves an assignment using the assignment's ID. Requesters can only
        retrieve their own assignments, and only assignments whose related HIT
        has not been disposed.

        The returned ResultSet will have the following attributes:

        Request
                This element is present only if the Request ResponseGroup
                is specified.
        Assignment
                The assignment. The response includes one Assignment object.
        HIT
                The HIT associated with this assignment. The response
                includes one HIT object.

        """
    params = {'AssignmentId': assignment_id}
    if response_groups:
        self.build_list_params(params, response_groups, 'ResponseGroup')
    return self._process_request('GetAssignment', params, [('Assignment', Assignment), ('HIT', HIT)])
from collections import abc
import xml.sax
import hashlib
import string
from boto.connection import AWSQueryConnection
from boto.exception import BotoServerError
import boto.mws.exception
import boto.mws.response
from boto.handler import XmlHandler
from boto.compat import filter, map, six, encodebytes
@requires(['NextToken'])
@api_action('Reports', 0, 0)
def get_report_request_list_by_next_token(self, request, response, **kw):
    """Returns a list of report requests using the NextToken,
           which was supplied by a previous request to either
           GetReportRequestListByNextToken or GetReportRequestList, where
           the value of HasNext was true in that previous request.
        """
    return self._post_request(request, kw, response)
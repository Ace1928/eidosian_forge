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
@structured_lists('ReportTypeList.Type')
@api_action('Reports', 10, 45)
def get_report_schedule_count(self, request, response, **kw):
    """Returns a count of order report requests that are scheduled
           to be submitted to Amazon MWS.
        """
    return self._post_request(request, kw, response)
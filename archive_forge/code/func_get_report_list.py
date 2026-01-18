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
@boolean_arguments('Acknowledged')
@structured_lists('ReportRequestIdList.Id', 'ReportTypeList.Type')
@api_action('Reports', 10, 60)
def get_report_list(self, request, response, **kw):
    """Returns a list of reports that were created in the previous
           90 days that match the query parameters.
        """
    return self._post_request(request, kw, response)
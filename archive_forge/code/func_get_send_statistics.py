import re
import base64
from boto.compat import six, urllib
from boto.connection import AWSAuthConnection
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
import boto
import boto.jsonresponse
from boto.ses import exceptions as ses_exceptions
def get_send_statistics(self):
    """Fetches the user's sending statistics. The result is a list of data
        points, representing the last two weeks of sending activity.

        Each data point in the list contains statistics for a 15-minute
        interval.

        :rtype: dict
        :returns: A GetSendStatisticsResponse structure. Note that keys must be
                  unicode strings.
        """
    return self._make_request('GetSendStatistics')
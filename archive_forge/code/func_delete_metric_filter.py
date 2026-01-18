import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.logs import exceptions
from boto.compat import json
def delete_metric_filter(self, log_group_name, filter_name):
    """
        Deletes a metric filter associated with the specified log
        group.

        :type log_group_name: string
        :param log_group_name:

        :type filter_name: string
        :param filter_name: The name of the metric filter.

        """
    params = {'logGroupName': log_group_name, 'filterName': filter_name}
    return self.make_request(action='DeleteMetricFilter', body=json.dumps(params))
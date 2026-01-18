import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.configservice import exceptions
def describe_delivery_channel_status(self, delivery_channel_names=None):
    """
        Returns the current status of the specified delivery channel.
        If a delivery channel is not specified, this action returns
        the current status of all delivery channels associated with
        the account.

        :type delivery_channel_names: list
        :param delivery_channel_names: A list of delivery channel names.

        """
    params = {}
    if delivery_channel_names is not None:
        params['DeliveryChannelNames'] = delivery_channel_names
    return self.make_request(action='DescribeDeliveryChannelStatus', body=json.dumps(params))
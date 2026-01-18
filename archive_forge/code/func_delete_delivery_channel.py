import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.configservice import exceptions
def delete_delivery_channel(self, delivery_channel_name):
    """
        Deletes the specified delivery channel.

        The delivery channel cannot be deleted if it is the only
        delivery channel and the configuration recorder is still
        running. To delete the delivery channel, stop the running
        configuration recorder using the StopConfigurationRecorder
        action.

        :type delivery_channel_name: string
        :param delivery_channel_name: The name of the delivery channel to
            delete.

        """
    params = {'DeliveryChannelName': delivery_channel_name}
    return self.make_request(action='DeleteDeliveryChannel', body=json.dumps(params))
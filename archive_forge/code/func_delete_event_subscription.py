import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def delete_event_subscription(self, subscription_name):
    """
        Deletes an Amazon Redshift event notification subscription.

        :type subscription_name: string
        :param subscription_name: The name of the Amazon Redshift event
            notification subscription to be deleted.

        """
    params = {'SubscriptionName': subscription_name}
    return self._make_request(action='DeleteEventSubscription', verb='POST', path='/', params=params)
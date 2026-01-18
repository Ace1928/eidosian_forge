import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.configservice import exceptions
def put_delivery_channel(self, delivery_channel):
    """
        Creates a new delivery channel object to deliver the
        configuration information to an Amazon S3 bucket, and to an
        Amazon SNS topic.

        You can use this action to change the Amazon S3 bucket or an
        Amazon SNS topic of the existing delivery channel. To change
        the Amazon S3 bucket or an Amazon SNS topic, call this action
        and specify the changed values for the S3 bucket and the SNS
        topic. If you specify a different value for either the S3
        bucket or the SNS topic, this action will keep the existing
        value for the parameter that is not changed.

        :type delivery_channel: dict
        :param delivery_channel: The configuration delivery channel object that
            delivers the configuration information to an Amazon S3 bucket, and
            to an Amazon SNS topic.

        """
    params = {'DeliveryChannel': delivery_channel}
    return self.make_request(action='PutDeliveryChannel', body=json.dumps(params))
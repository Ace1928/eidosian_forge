from boto.pyami.config import Config, BotoConfigLocations
from boto.storage_uri import BucketStorageUri, FileStorageUri
import boto.plugin
import datetime
import os
import platform
import re
import sys
import logging
import logging.config
from boto.compat import urlparse
from boto.exception import InvalidUriError
def connect_logs(aws_access_key_id=None, aws_secret_access_key=None, **kwargs):
    """
    Connect to Amazon CloudWatch Logs

    :type aws_access_key_id: string
    :param aws_access_key_id: Your AWS Access Key ID

    :type aws_secret_access_key: string
    :param aws_secret_access_key: Your AWS Secret Access Key

    rtype: :class:`boto.kinesis.layer1.CloudWatchLogsConnection`
    :return: A connection to the Amazon CloudWatch Logs service
    """
    from boto.logs.layer1 import CloudWatchLogsConnection
    return CloudWatchLogsConnection(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, **kwargs)
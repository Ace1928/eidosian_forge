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
def connect_directconnect(aws_access_key_id=None, aws_secret_access_key=None, **kwargs):
    """
    Connect to AWS DirectConnect

    :type aws_access_key_id: string
    :param aws_access_key_id: Your AWS Access Key ID

    :type aws_secret_access_key: string
    :param aws_secret_access_key: Your AWS Secret Access Key

    :rtype: :class:`boto.directconnect.layer1.DirectConnectConnection`
    :return: A connection to the AWS DirectConnect service
    """
    from boto.directconnect.layer1 import DirectConnectConnection
    return DirectConnectConnection(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, **kwargs)
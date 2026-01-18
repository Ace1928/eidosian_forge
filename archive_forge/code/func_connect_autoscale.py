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
def connect_autoscale(aws_access_key_id=None, aws_secret_access_key=None, **kwargs):
    """
    :type aws_access_key_id: string
    :param aws_access_key_id: Your AWS Access Key ID

    :type aws_secret_access_key: string
    :param aws_secret_access_key: Your AWS Secret Access Key

    :rtype: :class:`boto.ec2.autoscale.AutoScaleConnection`
    :return: A connection to Amazon's Auto Scaling Service

    :type use_block_device_types bool
    :param use_block_device_types: Specifies whether to return described Launch Configs with block device mappings containing
        block device types, or a list of old style block device mappings (deprecated).  This defaults to false for compatability
        with the old incorrect style.
    """
    from boto.ec2.autoscale import AutoScaleConnection
    return AutoScaleConnection(aws_access_key_id, aws_secret_access_key, **kwargs)
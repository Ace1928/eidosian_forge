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
def connect_opsworks(aws_access_key_id=None, aws_secret_access_key=None, **kwargs):
    from boto.opsworks.layer1 import OpsWorksConnection
    return OpsWorksConnection(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, **kwargs)
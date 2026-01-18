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
def connect_gs(gs_access_key_id=None, gs_secret_access_key=None, **kwargs):
    """
    @type gs_access_key_id: string
    @param gs_access_key_id: Your Google Cloud Storage Access Key ID

    @type gs_secret_access_key: string
    @param gs_secret_access_key: Your Google Cloud Storage Secret Access Key

    @rtype: L{GSConnection<boto.gs.connection.GSConnection>}
    @return: A connection to Google's Storage service
    """
    from boto.gs.connection import GSConnection
    return GSConnection(gs_access_key_id, gs_secret_access_key, **kwargs)
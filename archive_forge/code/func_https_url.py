import logging
import os
import urllib.parse
from oslo_config import cfg
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import units
import requests
from requests import adapters
from requests.packages.urllib3.util import retry
import glance_store
from glance_store import capabilities
from glance_store.common import utils
from glance_store import exceptions
from glance_store.i18n import _, _LE
from glance_store import location
@property
def https_url(self):
    """
        Creates a https url that can be used to upload/download data from a
        vmware store.
        """
    parsed_url = urllib.parse.urlparse(self.get_uri())
    new_url = parsed_url._replace(scheme='https')
    return urllib.parse.urlunparse(new_url)
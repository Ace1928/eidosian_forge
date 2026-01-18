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
def new_session(insecure=False, ca_file=None, total_retries=None):
    session = requests.Session()
    if total_retries is not None:
        http_adapter = adapters.HTTPAdapter(max_retries=retry.Retry(total=total_retries))
        https_adapter = adapters.HTTPAdapter(max_retries=retry.Retry(total=total_retries))
        session.mount('http://', http_adapter)
        session.mount('https://', https_adapter)
    session.verify = ca_file if ca_file else not insecure
    return session
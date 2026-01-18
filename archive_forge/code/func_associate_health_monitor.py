import inspect
import itertools
import logging
import re
import time
import urllib.parse as urlparse
import debtcollector.renames
from keystoneauth1 import exceptions as ksa_exc
import requests
from neutronclient._i18n import _
from neutronclient import client
from neutronclient.common import exceptions
from neutronclient.common import extension as client_extension
from neutronclient.common import serializer
from neutronclient.common import utils
def associate_health_monitor(self, pool, body):
    """Associate  specified load balancer health monitor and pool."""
    return self.post(self.associate_pool_health_monitors_path % pool, body=body)
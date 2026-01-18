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
def get_lbaas_agent_hosting_pool(self, pool, **_params):
    """Fetches a loadbalancer agent hosting a pool."""
    return self.get((self.pool_path + self.LOADBALANCER_AGENT) % pool, params=_params)
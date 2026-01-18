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
def onboard_network_subnets(self, subnetpool, body=None):
    """Onboard the specified network's subnets into a subnet pool."""
    return self.put(self.onboard_network_subnets_path % subnetpool, body=body)
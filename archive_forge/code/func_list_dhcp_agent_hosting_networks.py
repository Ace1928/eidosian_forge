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
def list_dhcp_agent_hosting_networks(self, network, **_params):
    """Fetches a list of dhcp agents hosting a network."""
    return self.get((self.network_path + self.DHCP_AGENTS) % network, params=_params)
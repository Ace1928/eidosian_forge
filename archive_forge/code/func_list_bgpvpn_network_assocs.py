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
def list_bgpvpn_network_assocs(self, bgpvpn, retrieve_all=True, **_params):
    """Fetches a list of network associations for a given BGP VPN."""
    return self.list('network_associations', self.bgpvpn_network_associations_path % bgpvpn, retrieve_all, **_params)
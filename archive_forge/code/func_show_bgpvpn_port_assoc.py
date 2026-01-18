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
def show_bgpvpn_port_assoc(self, bgpvpn, port_assoc, **_params):
    """Fetches information of a certain BGP VPN's port association"""
    return self.get(self.bgpvpn_port_association_path % (bgpvpn, port_assoc), params=_params)
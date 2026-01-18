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
def show_network_ip_availability(self, network, **_params):
    """Fetches IP availability information for a specified network"""
    return self.get(self.network_ip_availability_path % network, params=_params)
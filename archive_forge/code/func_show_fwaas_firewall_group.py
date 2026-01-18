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
def show_fwaas_firewall_group(self, fwg, **_params):
    """Fetches information of a certain firewall group"""
    return self.get(self.fwaas_firewall_group_path % fwg, params=_params)
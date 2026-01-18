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
def list_dragents_hosting_bgp_speaker(self, bgp_speaker, **_params):
    """Fetches a list of Dynamic Routing agents hosting a BGP speaker."""
    return self.get((self.bgp_speaker_path + self.BGP_DRAGENTS) % bgp_speaker, params=_params)
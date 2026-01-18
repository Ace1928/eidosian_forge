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
def show_bgp_speaker(self, bgp_speaker_id, **_params):
    """Fetches information of a certain BGP speaker."""
    return self.get(self.bgp_speaker_path % bgp_speaker_id, params=_params)
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
def list_l3_agent_hosting_routers(self, router, **_params):
    """Fetches a list of L3 agents hosting a router."""
    return self.get((self.router_path + self.L3_AGENTS) % router, params=_params)
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
def list_routers_on_l3_agent(self, l3_agent, **_params):
    """Fetches a list of routers hosted on an L3 agent."""
    return self.get((self.agent_path + self.L3_ROUTERS) % l3_agent, params=_params)
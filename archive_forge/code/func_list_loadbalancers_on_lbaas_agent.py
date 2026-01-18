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
def list_loadbalancers_on_lbaas_agent(self, lbaas_agent, **_params):
    """Fetches a list of loadbalancers hosted by the loadbalancer agent."""
    return self.get((self.agent_path + self.AGENT_LOADBALANCERS) % lbaas_agent, params=_params)
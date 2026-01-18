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
def add_router_to_l3_agent(self, l3_agent, body):
    """Adds a router to L3 agent."""
    return self.post((self.agent_path + self.L3_ROUTERS) % l3_agent, body=body)
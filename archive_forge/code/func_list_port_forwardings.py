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
def list_port_forwardings(self, floatingip, retrieve_all=True, **_params):
    """Fetches a list of all portforwardings for a floatingip."""
    return self.list('port_forwardings', self.port_forwardings_path % floatingip, retrieve_all, **_params)
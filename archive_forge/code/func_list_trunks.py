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
def list_trunks(self, retrieve_all=True, **_params):
    """Fetch a list of all trunk ports."""
    return self.list('trunks', self.trunks_path, retrieve_all, **_params)
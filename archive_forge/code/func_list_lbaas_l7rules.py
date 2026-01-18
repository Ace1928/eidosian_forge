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
def list_lbaas_l7rules(self, l7policy, retrieve_all=True, **_params):
    """Fetches a list of all rules for L7 policy."""
    return self.list('rules', self.lbaas_l7rules_path % l7policy, retrieve_all, **_params)
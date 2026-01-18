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
def list_minimum_packet_rate_rules(self, policy_id, retrieve_all=True, **_params):
    """Fetches a list of all minimum packet rate rules for the given policy

        """
    return self.list('minimum_packet_rate_rules', self.qos_minimum_packet_rate_rules_path % policy_id, retrieve_all, **_params)
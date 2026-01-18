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
def remove_rule_fwaas_firewall_policy(self, firewall_policy, body=None):
    """Removes specified rule from firewall policy"""
    return self.put(self.fwaas_firewall_policy_remove_path % firewall_policy, body=body)
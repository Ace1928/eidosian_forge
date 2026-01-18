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
def show_security_group_rule(self, security_group_rule, **_params):
    """Fetches information of a certain security group rule."""
    return self.get(self.security_group_rule_path % security_group_rule, params=_params)
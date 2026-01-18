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
def show_sfc_flow_classifier(self, flow_classifier, **_params):
    """Fetches information of a certain Flow Classifier."""
    return self.get(self.sfc_flow_classifier_path % flow_classifier, params=_params)
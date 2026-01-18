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
def extend_create(self, resource_singular, path, parent_resource):

    def _fx(body=None):
        return self.create_ext(path, body)

    def _parent_fx(parent_id, body=None):
        return self.create_ext(path % parent_id, body)
    fn = _fx if not parent_resource else _parent_fx
    setattr(self, 'create_%s' % resource_singular, fn)
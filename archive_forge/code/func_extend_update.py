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
def extend_update(self, resource_singular, path, parent_resource):

    def _fx(obj, body=None):
        return self.update_ext(path, obj, body)

    def _parent_fx(obj, parent_id, body=None):
        return self.update_ext(path % parent_id, obj, body)
    fn = _fx if not parent_resource else _parent_fx
    setattr(self, 'update_%s' % resource_singular, fn)
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
def extend_show(self, resource_singular, path, parent_resource):

    def _fx(obj, **_params):
        return self.show_ext(path, obj, **_params)

    def _parent_fx(obj, parent_id, **_params):
        return self.show_ext(path % parent_id, obj, **_params)
    fn = _fx if not parent_resource else _parent_fx
    setattr(self, 'show_%s' % resource_singular, fn)
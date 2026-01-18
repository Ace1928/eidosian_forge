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
def extend_delete(self, resource_singular, path, parent_resource):

    def _fx(obj):
        return self.delete_ext(path, obj)

    def _parent_fx(obj, parent_id):
        return self.delete_ext(path % parent_id, obj)
    fn = _fx if not parent_resource else _parent_fx
    setattr(self, 'delete_%s' % resource_singular, fn)
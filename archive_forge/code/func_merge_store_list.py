import copy
import functools
import json
import os
import urllib.request
import glance_store as store_api
from glance_store import backend
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import timeutils
from oslo_utils import units
import taskflow
from taskflow.patterns import linear_flow as lf
from taskflow import retry
from taskflow import task
from glance.api import common as api_common
import glance.async_.flows._internal_plugins as internal_plugins
import glance.async_.flows.plugins as import_plugins
from glance.async_ import utils
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance.common.scripts import utils as script_utils
from glance.common import store_utils
from glance.i18n import _, _LE, _LI
from glance.quota import keystone as ks_quota
def merge_store_list(self, list_key, stores, subtract=False):
    stores = set([store for store in stores if store])
    existing = set(self._image.extra_properties.get(list_key, '').split(','))
    if subtract:
        if stores - existing:
            LOG.debug('Stores %(stores)s not in %(key)s for image %(image_id)s', {'stores': ','.join(sorted(stores - existing)), 'key': list_key, 'image_id': self.image_id})
        merged_stores = existing - stores
    else:
        merged_stores = existing | stores
    stores_list = ','.join(sorted((store for store in merged_stores if store)))
    self._image.extra_properties[list_key] = stores_list
    LOG.debug('Image %(image_id)s %(key)s=%(stores)s', {'image_id': self.image_id, 'key': list_key, 'stores': stores_list})
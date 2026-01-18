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
def remove_location_for_store(self, backend):
    """Remove a location from an image given a backend store.

        Given a backend store, remove the corresponding location from the
        image's set of locations. If the last location is removed, remove
        the image checksum, hash information, and size.

        :param backend: The backend store to remove from the image
        """
    for i, location in enumerate(self._image.locations):
        if location.get('metadata', {}).get('store') == backend:
            try:
                self._image.locations.pop(i)
            except (store_exceptions.NotFound, store_exceptions.Forbidden):
                msg = _('Error deleting from store %(store)s when reverting.') % {'store': backend}
                LOG.warning(msg)
            except Exception:
                msg = _('Unexpected exception when deleting from store %(store)s.') % {'store': backend}
                LOG.warning(msg)
            else:
                if len(self._image.locations) == 0:
                    self._image.checksum = None
                    self._image.os_hash_algo = None
                    self._image.os_hash_value = None
                    self._image.size = None
            break
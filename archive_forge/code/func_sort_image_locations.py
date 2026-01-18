import errno
from eventlet.green import socket
import functools
import os
import re
import urllib
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import strutils
from webob import exc
from glance.common import exception
from glance.common import location_strategy
from glance.common import timeutils
from glance.common import wsgi
from glance.i18n import _, _LE, _LW
def sort_image_locations(locations):
    if not CONF.enabled_backends:
        return location_strategy.get_ordered_locations(locations)

    def get_store_weight(location):
        store_id = location['metadata'].get('store')
        if not store_id:
            return 0
        try:
            store = glance_store.get_store_from_store_identifier(store_id)
        except glance_store.exceptions.UnknownScheme:
            msg = _LW("Unable to find store '%s', returning default weight '0'") % store_id
            LOG.warning(msg)
            return 0
        return store.weight if store is not None else 0
    sorted_locations = sorted(locations, key=get_store_weight, reverse=True)
    LOG.debug('Sorted locations: %s', sorted_locations)
    return sorted_locations
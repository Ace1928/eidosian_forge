import copy
import hashlib
import logging
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import units
from stevedore import driver
from stevedore import extension
from glance_store import capabilities
from glance_store import exceptions
from glance_store.i18n import _
from glance_store import location
def get_store_from_store_identifier(store_identifier):
    """Determine backing store from identifier.

    Given a store identifier, return the appropriate store object
    for handling that scheme.
    """
    scheme_map = {}
    enabled_backends = CONF.enabled_backends
    enabled_backends.update(_RESERVED_STORES)
    try:
        scheme = enabled_backends[store_identifier]
    except KeyError:
        msg = _('Store for identifier %s not found') % store_identifier
        raise exceptions.UnknownScheme(msg)
    if scheme not in location.SCHEME_TO_CLS_BACKEND_MAP:
        raise exceptions.UnknownScheme(scheme=scheme)
    scheme_info = location.SCHEME_TO_CLS_BACKEND_MAP[scheme][store_identifier]
    store = scheme_info['store']
    if not store.is_capable(capabilities.BitMasks.DRIVER_REUSABLE):
        store_entry = scheme_info['store_entry']
        store = _load_multi_store(store.conf, store_entry, invoke_load=True, backend=store_identifier)
        store.configure()
        try:
            loc_cls = store.get_store_location_class()
            for new_scheme in store.get_schemes():
                if new_scheme not in scheme_map:
                    scheme_map[new_scheme] = {}
                scheme_map[new_scheme][store_identifier] = {'store': store, 'location_class': loc_cls, 'store_entry': store_entry}
                location.register_scheme_backend_map(scheme_map)
        except NotImplementedError:
            scheme_info['store'] = store
    return store
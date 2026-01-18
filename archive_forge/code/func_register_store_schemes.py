import os
import shutil
import fixtures
from oslo_config import cfg
from oslotest import base
import glance_store as store
from glance_store import location
def register_store_schemes(self, store, store_entry):
    schemes = store.get_schemes()
    scheme_map = {}
    loc_cls = store.get_store_location_class()
    for scheme in schemes:
        scheme_map[scheme] = {'store': store, 'location_class': loc_cls, 'store_entry': store_entry}
    location.register_scheme_map(scheme_map)
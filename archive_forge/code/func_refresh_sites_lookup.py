from __future__ import absolute_import, division, print_function
import json
import uuid
import math
import os
import datetime
from copy import deepcopy
from functools import partial
from sys import version as python_version
from threading import Thread
from typing import Iterable
from itertools import chain
from collections import defaultdict
from ipaddress import ip_interface
from ansible.constants import DEFAULT_LOCAL_TMP
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib import error as urllib_error
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.six import raise_from
def refresh_sites_lookup(self):
    url = self.api_endpoint + '/api/dcim/sites/?limit=0'
    sites = self.get_resource_list(api_url=url)
    self.sites_lookup_slug = dict(((site['id'], site['slug']) for site in sites))
    if self.site_data or self.prefixes:
        self.sites_lookup = dict(((site['id'], site) for site in sites))
    else:
        self.sites_lookup = self.sites_lookup_slug
    self.sites_with_prefixes = set()
    for site in sites:
        if site['prefix_count'] and site['prefix_count'] > 0:
            self.sites_with_prefixes.add(site['slug'])

    def get_region_for_site(site):
        try:
            return (site['id'], site['region']['id'])
        except Exception:
            return (site['id'], None)
    self.sites_region_lookup = dict(map(get_region_for_site, sites))

    def get_site_group_for_site(site):
        try:
            return (site['id'], site['group']['id'])
        except Exception:
            return (site['id'], None)
    self.sites_site_group_lookup = dict(map(get_site_group_for_site, sites))

    def get_time_zone_for_site(site):
        try:
            return (site['id'], site['time_zone'].replace('/', '_', 2))
        except Exception:
            return (site['id'], None)
    if 'time_zone' in self.group_by:
        self.sites_time_zone_lookup = dict(map(get_time_zone_for_site, sites))

    def get_utc_offset_for_site(site):
        try:
            utc = round(datetime.datetime.now(pytz.timezone(site['time_zone'])).utcoffset().total_seconds() / 60 / 60)
            if utc < 0:
                return (site['id'], str(utc).replace('-', 'minus_'))
            else:
                return (site['id'], f'plus_{utc}')
        except Exception:
            return (site['id'], None)
    if 'utc_offset' in self.group_by:
        self.sites_utc_offset_lookup = dict(map(get_utc_offset_for_site, sites))

    def get_facility_for_site(site):
        try:
            return (site['id'], site['facility'])
        except Exception:
            return (site['id'], None)
    if 'facility' in self.group_by:
        self.sites_facility_lookup = dict(map(get_facility_for_site, sites))
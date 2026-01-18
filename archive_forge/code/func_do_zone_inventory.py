from __future__ import (absolute_import, division, print_function)
import os
import json
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible_collections.community.general.plugins.module_utils.scaleway import SCALEWAY_LOCATION, parse_pagination_link
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.six import raise_from
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
import ansible.module_utils.six.moves.urllib.parse as urllib_parse
def do_zone_inventory(self, zone, token, tags, hostname_preferences):
    self.inventory.add_group(zone)
    zone_info = SCALEWAY_LOCATION[zone]
    url = _build_server_url(zone_info['api_endpoint'])
    raw_zone_hosts_infos = make_unsafe(_fetch_information(url=url, token=token))
    for host_infos in raw_zone_hosts_infos:
        hostname = self._filter_host(host_infos=host_infos, hostname_preferences=hostname_preferences)
        if not hostname:
            continue
        groups = self.match_groups(host_infos, tags)
        for group in groups:
            self.inventory.add_group(group=group)
            self.inventory.add_host(group=group, host=hostname)
            self._fill_host_variables(host=hostname, server_info=host_infos)
            self._set_composite_vars(self.get_option('variables'), host_infos, hostname, strict=False)
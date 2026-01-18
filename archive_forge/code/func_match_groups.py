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
def match_groups(self, server_info, tags):
    server_zone = extract_zone(server_info=server_info)
    server_tags = extract_tags(server_info=server_info)
    if server_zone is None:
        return set()
    if tags is None:
        return set(server_tags).union((server_zone,))
    matching_tags = set(server_tags).intersection(tags)
    if not matching_tags:
        return set()
    return matching_tags.union((server_zone,))
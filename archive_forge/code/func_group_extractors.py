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
@property
def group_extractors(self):
    extractors = {'disk': self.extract_disk, 'memory': self.extract_memory, 'vcpus': self.extract_vcpus, 'status': self.extract_status, 'config_context': self.extract_config_context, 'local_context_data': self.extract_local_context_data, 'custom_fields': self.extract_custom_fields, 'region': self.extract_regions, 'cluster': self.extract_cluster, 'cluster_group': self.extract_cluster_group, 'cluster_type': self.extract_cluster_type, 'is_virtual': self.extract_is_virtual, 'serial': self.extract_serial, 'asset_tag': self.extract_asset_tag, 'time_zone': self.extract_site_time_zone, 'utc_offset': self.extract_site_utc_offset, 'facility': self.extract_site_facility, self._pluralize_group_by('site'): self.extract_site, self._pluralize_group_by('tenant'): self.extract_tenant, self._pluralize_group_by('tag'): self.extract_tags, self._pluralize_group_by('role'): self.extract_device_role, self._pluralize_group_by('platform'): self.extract_platform, self._pluralize_group_by('device_type'): self.extract_device_type, self._pluralize_group_by('manufacturer'): self.extract_manufacturer}
    if self.api_version >= version.parse('2.11'):
        extractors.update({'site_group': self.extract_site_groups})
    if self.racks:
        extractors.update({self._pluralize_group_by('rack'): self.extract_rack, 'rack_role': self.extract_rack_role})
        if self.api_version >= version.parse('2.11'):
            extractors.update({'location': self.extract_location})
        else:
            extractors.update({'rack_group': self.extract_rack_group})
    if self.services:
        extractors.update({'services': self.extract_services})
    if self.interfaces:
        extractors.update({'interfaces': self.extract_interfaces})
    if self.interfaces or self.dns_name or self.ansible_host_dns_name:
        extractors.update({'dns_name': self.extract_dns_name})
    return extractors
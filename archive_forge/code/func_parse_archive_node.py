from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.logging_global.logging_global import (
def parse_archive_node(self, conf):
    archive_dict = {}
    if conf is not None:
        if 'binary-data' in conf.keys():
            archive_dict['binary_data'] = True
        if 'files' in conf.keys():
            archive_dict['files'] = conf.get('files')
        if 'no-binary-data' in conf.keys():
            archive_dict['no_binary_data'] = True
        if 'no-world-readable' in conf.keys():
            archive_dict['no_world_readable'] = True
        if 'size' in conf.keys():
            archive_dict['file_size'] = conf.get('size')
        if 'world-readable' in conf.keys():
            archive_dict['world_readable'] = True
        if 'archive-sites' in conf.keys():
            archive_sites_list = []
            archive_sites = conf.get('archive-sites')
            if isinstance(archive_sites, list):
                for item in archive_sites:
                    archive_sites_list.append(item['name'])
            else:
                archive_sites_list.append(archive_sites['name'])
            archive_dict['archive_sites'] = archive_sites_list
    else:
        archive_dict['set'] = True
    return archive_dict
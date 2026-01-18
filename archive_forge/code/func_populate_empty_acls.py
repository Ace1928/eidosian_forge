from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.acls.acls import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.acls import (
def populate_empty_acls(self, raw_acls, raw_acls_name):
    if raw_acls and raw_acls_name:
        for aclnames, acldata in raw_acls_name.get('acls').items():
            if aclnames not in raw_acls.get('acls').keys():
                if not raw_acls.get('acls'):
                    raw_acls['acls'] = {}
                raw_acls['acls'][aclnames] = acldata
    elif raw_acls_name and (not raw_acls):
        for aclnames, acldata in raw_acls_name.get('acls').items():
            if not raw_acls.get('acls'):
                raw_acls['acls'] = {}
            raw_acls['acls'][aclnames] = acldata
    return raw_acls
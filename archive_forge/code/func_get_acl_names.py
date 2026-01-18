from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.acls.acls import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.acls import (
def get_acl_names(self, connection):
    return connection.get('show access-lists | include access list')
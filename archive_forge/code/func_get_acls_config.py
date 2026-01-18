from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.argspec.acls.acls import (
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.rm_templates.acls import (
def get_acls_config(self, connection):
    return connection.get('sh access-list')
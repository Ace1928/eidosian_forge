from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.argspec.ogs.ogs import OGsArgs
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.rm_templates.ogs import (
def get_og_data(self, connection):
    return connection.get('sh running-config object-group')
from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def get_route_type(self, dest=None, afi=None):
    """
        This function returns the route type based on
        destination ip address or afi
        :param address:
        :return:
        """
    if dest:
        return get_route_type(dest)
    elif afi == 'ipv4':
        return 'route'
    elif afi == 'ipv6':
        return 'route6'
from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def search_route_in_have(self, have, want_dest):
    """
        This function  returns the route if its found in
        have config.
        :param have:
        :param dest:
        :return: the matched route
        """
    routes = self._get_routes(have)
    for r in routes:
        if r['dest'] == want_dest:
            return r
    return None
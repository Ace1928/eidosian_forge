from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
def find_af_context(self, want_af_context, have_address_families):
    """This method iterates through the have AFs
            and returns the one that matches the want AF

        :rtype: A dict
        :returns: the corresponding AF in have AFs
                  that matches the want AF
        """
    for have_af in have_address_families:
        if have_af['afi'] == want_af_context['afi'] and have_af['safi'] == want_af_context['safi']:
            return have_af
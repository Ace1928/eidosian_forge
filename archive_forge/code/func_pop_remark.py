from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.acls import (
def pop_remark(r_entry, afi):
    """Takes out remarks from ace entry as remarks not same
            does not mean the ace entry to be re-introduced
            """
    if r_entry.get('remarks'):
        return r_entry.pop('remarks')
    else:
        return {}
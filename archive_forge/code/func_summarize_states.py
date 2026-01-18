from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def summarize_states(self):
    """ replaces a long list of states with multipliers
            eg 'false*5' or 'false*2,true'
            return:
                state_list as str
                last_state
        """
    previous_state = None
    count = 0
    summaries = []
    for state in self.states:
        if state == previous_state:
            count += 1
        else:
            if previous_state is not None:
                summaries.append('%s%s' % (previous_state, '' if count == 1 else '*%d' % count))
            count = 1
            previous_state = state
    if previous_state is not None:
        summaries.append('%s%s' % (previous_state, '' if count == 1 else '*%d' % count))
    last_state = self.states[-1] if self.states else ''
    return (','.join(summaries), last_state)
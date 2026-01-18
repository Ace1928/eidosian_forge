from __future__ import (absolute_import, division, print_function)
import json
import os
from functools import partial
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.validation import check_type_dict, safe_eval
def check_if_add_remove_ip_arg_exists(self, proposed_object):
    """
            This function shall check if add/remove param is set to true and
            is passed in the args, then we will update the proposed dictionary
            to add/remove IP to existing host_record, if the user passes false
            param with the argument nothing shall be done.
            :returns: True if param is changed based on add/remove, and also the
            changed proposed_object.
        """
    update = False
    if 'add' in proposed_object['ipv4addrs'][0]:
        if proposed_object['ipv4addrs'][0]['add']:
            proposed_object['ipv4addrs+'] = proposed_object['ipv4addrs']
            del proposed_object['ipv4addrs']
            del proposed_object['ipv4addrs+'][0]['add']
            update = True
        else:
            del proposed_object['ipv4addrs'][0]['add']
    elif 'remove' in proposed_object['ipv4addrs'][0]:
        if proposed_object['ipv4addrs'][0]['remove']:
            proposed_object['ipv4addrs-'] = proposed_object['ipv4addrs']
            del proposed_object['ipv4addrs']
            del proposed_object['ipv4addrs-'][0]['remove']
            update = True
        else:
            del proposed_object['ipv4addrs'][0]['remove']
    return (update, proposed_object)
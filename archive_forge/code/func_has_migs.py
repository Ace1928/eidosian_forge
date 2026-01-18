from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def has_migs(self, local=True):
    """returns a boolean, False if no migrations otherwise True"""
    consecutive_good = 0
    try_num = 0
    skip_reason = list()
    while try_num < int(self.module.params['tries_limit']) and consecutive_good < int(self.module.params['consecutive_good_checks']):
        self._update_nodes_list()
        self._update_cluster_statistics()
        stable, reason = self._cluster_good_state()
        if stable is not True:
            skip_reason.append('Skipping on try#' + str(try_num) + ' for reason:' + reason)
        elif self._can_use_cluster_stable():
            if self._cluster_stable():
                consecutive_good += 1
            else:
                consecutive_good = 0
                skip_reason.append('Skipping on try#' + str(try_num) + ' for reason:' + ' cluster_stable')
        elif self._has_migs(local):
            skip_reason.append('Skipping on try#' + str(try_num) + ' for reason:' + ' migrations')
            consecutive_good = 0
        else:
            consecutive_good += 1
            if consecutive_good == self.module.params['consecutive_good_checks']:
                break
        try_num += 1
        sleep(self.module.params['sleep_between_checks'])
    if consecutive_good == self.module.params['consecutive_good_checks']:
        return (False, None)
    return (True, skip_reason)
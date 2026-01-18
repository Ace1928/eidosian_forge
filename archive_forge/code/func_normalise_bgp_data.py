from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.bgp.bgp import BgpArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
def normalise_bgp_data(self, data):
    for conf in data:
        bestpath = {}
        med = {}
        timers = {}
        as_path = {}
        max_med_on_start_up = {}
        conf['log_neighbor_changes'] = conf.get('log_neighbor_changes', False)
        as_path['confed'] = conf.get('as_path_confed', False)
        as_path['ignore'] = conf.get('as_path_ignore', False)
        as_path['multipath_relax'] = conf.get('as_path_multipath_relax', False)
        as_path['multipath_relax_as_set'] = conf.get('as_path_multipath_relax_as_set', False)
        bestpath['as_path'] = as_path
        med['confed'] = conf.get('med_confed', False)
        med['missing_as_worst'] = conf.get('med_missing_as_worst', False)
        med['always_compare_med'] = conf.get('always_compare_med', False)
        bestpath['med'] = med
        timers['holdtime'] = conf.get('holdtime', None)
        timers['keepalive_interval'] = conf.get('keepalive_interval', None)
        conf['timers'] = timers
        bestpath['compare_routerid'] = conf.get('compare_routerid', False)
        conf['bestpath'] = bestpath
        max_med_on_start_up['timer'] = conf.get('max_med_on_startup_timer', None)
        max_med_on_start_up['med_val'] = conf.get('max_med_on_startup_med_val', None)
        conf['max_med'] = {'on_startup': max_med_on_start_up}
        keys = ['as_path_confed', 'as_path_ignore', 'as_path_multipath_relax', 'as_path_multipath_relax_as_set', 'med_confed', 'med_missing_as_worst', 'always_compare_med', 'max_med_val', 'holdtime', 'keepalive_interval', 'compare_routerid', 'admin_max_med', 'max_med_on_startup_timer', 'max_med_on_startup_med_val']
        for key in keys:
            if key in conf:
                conf.pop(key)
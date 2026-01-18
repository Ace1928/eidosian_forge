from __future__ import absolute_import, division, print_function
import os
import re
import platform
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def create_snap_policy(module, blade):
    """Create REST 2 snapshot policy"""
    changed = True
    if module.params['keep_for'] and (not module.params['every']) or (module.params['every'] and (not module.params['keep_for'])):
        module.fail_json(msg='`keep_for` and `every` are required.')
    if module.params['timezone'] and (not module.params['at']):
        module.fail_json(msg='`timezone` requires `at` to be provided.')
    if module.params['at'] and (not module.params['every']):
        module.fail_json(msg='`at` requires `every` to be provided.')
    if not module.check_mode:
        if module.params['at'] and module.params['every']:
            if not module.params['every'] % 86400 == 0:
                module.fail_json(msg='At time can only be set if every value is a multiple of 86400')
            if not module.params['timezone']:
                module.params['timezone'] = _get_local_tz(module)
                if module.params['timezone'] not in pytz.all_timezones_set:
                    module.fail_json(msg='Timezone {0} is not valid'.format(module.params['timezone']))
        if not module.params['keep_for']:
            module.params['keep_for'] = 0
        if not module.params['every']:
            module.params['every'] = 0
        if module.params['keep_for'] < module.params['every']:
            module.fail_json(msg='Retention period cannot be less than snapshot interval.')
        if module.params['at'] and (not module.params['timezone']):
            module.params['timezone'] = _get_local_tz(module)
            if module.params['timezone'] not in set(pytz.all_timezones_set):
                module.fail_json(msg='Timezone {0} is not valid'.format(module.params['timezone']))
        if module.params['keep_for']:
            if not 300 <= module.params['keep_for'] <= 34560000:
                module.fail_json(msg='keep_for parameter is out of range (300 to 34560000)')
            if not 300 <= module.params['every'] <= 34560000:
                module.fail_json(msg='every parameter is out of range (300 to 34560000)')
            if module.params['at']:
                attr = Policy(enabled=module.params['enabled'], rules=[PolicyRule(keep_for=module.params['keep_for'] * 1000, every=module.params['every'] * 1000, at=_convert_to_millisecs(module.params['at']), time_zone=module.params['timezone'])])
            else:
                attr = Policy(enabled=module.params['enabled'], rules=[PolicyRule(keep_for=module.params['keep_for'] * 1000, every=module.params['every'] * 1000)])
        else:
            attr = Policy(enabled=module.params['enabled'])
        res = blade.post_policies(names=[module.params['name']], policy=attr)
        if res.status_code != 200:
            module.fail_json(msg='Failed to create snapshot policy {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)
from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import to_native
from ansible_collections.netapp.azure.plugins.module_utils.azure_rm_netapp_common import AzureRMNetAppModuleBase
from ansible_collections.netapp.azure.plugins.module_utils.netapp_module import NetAppModule
def get_changes_in_ads(self, current, desired):
    c_ads = current.get('active_directories')
    d_ads = desired.get('active_directories')
    if not c_ads:
        return (desired.get('active_directories'), None)
    if not d_ads:
        return (None, current.get('active_directories'))
    if len(c_ads) > 1 or len(d_ads) > 1:
        msg = 'Error checking for AD, currently only one AD is supported.'
        if len(c_ads) > 1:
            msg += '  Current: %s.' % str(c_ads)
        if len(d_ads) > 1:
            msg += '  Desired: %s.' % str(d_ads)
        self.module.fail_json(msg='Error checking for AD, currently only one AD is supported')
    changed = False
    d_ad = d_ads[0]
    c_ad = c_ads[0]
    for key, value in c_ad.items():
        if key == 'password':
            if d_ad.get(key) is None:
                continue
            self.warnings.append("module is not idempotent if 'password:' is present")
        if d_ad.get(key) is None:
            d_ad[key] = value
        elif d_ad.get(key) != value:
            changed = True
            self.debug.append('key: %s, value %s' % (key, value))
    if changed:
        return ([d_ad], None)
    return (None, None)
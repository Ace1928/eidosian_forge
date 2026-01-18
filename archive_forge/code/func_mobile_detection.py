from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def mobile_detection(self):
    tmp = dict()
    tmp['enabled'] = flatten_boolean(self._values['enable_mobile_detection'])
    tmp['allow_android_rooted_device'] = flatten_boolean(self._values['allow_android_rooted_device'])
    tmp['allow_any_android_package'] = flatten_boolean(self._values['allow_any_android_package'])
    tmp['allow_any_ios_package'] = flatten_boolean(self._values['allow_any_ios_package'])
    tmp['allow_jailbroken_devices'] = flatten_boolean(self._values['allow_jailbroken_devices'])
    tmp['allow_emulators'] = flatten_boolean(self._values['allow_emulators'])
    tmp['client_side_challenge_mode'] = self._values['client_side_challenge_mode']
    tmp['ios_allowed_package_names'] = self._values['ios_allowed_package_names']
    tmp['android_publishers'] = self._values['android_publishers']
    result = self._filter_params(tmp)
    if result:
        return result
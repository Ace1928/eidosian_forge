from __future__ import absolute_import, division, print_function
import time
import json
from ansible.plugins.httpapi import HttpApiBase
from ansible.module_utils.basic import to_text
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import BASE_HEADERS
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGBaseException
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGRCommon
from datetime import datetime
def get_locked_adom_list(self):
    """
        Gets the list of locked adoms
        """
    try:
        locked_list = list()
        locked_by_user_list = dict()
        for adom in self._adom_list:
            self.log('lockinfo for adom:%s' % adom)
            rc, adom_lock_info = self.get_lock_info(adom=adom)
            if adom_lock_info['status']['code'] != 0:
                continue
            if 'data' not in adom_lock_info:
                continue
            lock_data = adom_lock_info['data']
            if isinstance(lock_data, list):
                lock_data = lock_data[0]
            locked_list.append(adom)
            locked_by_user_list[adom] = lock_data['lock_user']
        self._locked_adom_list = locked_list
        self._locked_adoms_by_user = locked_by_user_list
        self.log('locked adom list: %s' % self._locked_adom_list)
        self.log('locked adom and user list: %s' % self._locked_adoms_by_user)
    except Exception as err:
        raise FMGBaseException(msg='An error occurred while trying to get the locked adom list. Error: ' + to_text(err))
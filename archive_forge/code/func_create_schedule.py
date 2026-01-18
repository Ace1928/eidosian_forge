from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def create_schedule(self, check_mode, result, cursor):
    if not check_mode:
        result['changed'] = self.create_schedule_config(cursor)
        result['msg'] = 'Added schedule to scheduler'
        result['schedules'] = self.get_schedule_config(cursor)
        self.manage_config(cursor, result['changed'])
    else:
        result['changed'] = True
        result['msg'] = 'Schedule would have been added to' + ' scheduler, however check_mode' + ' is enabled.'
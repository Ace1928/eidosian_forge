from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_replication_session_on_filter(self, obj_nas, replication_params, action):
    """ Retrieves replication session on nas server
            :param: obj_nas: NAS server object
            :param: replication_params: Module input params
            :param: action: Specifies action as modify or delete
            :return: Replication session based on filter
        """
    if replication_params and replication_params['remote_system']:
        repl_session = self.get_replication_session(obj_nas, filter_key='remote_system_name', replication_params=replication_params)
    elif replication_params and replication_params['replication_name']:
        repl_session = self.get_replication_session(obj_nas, filter_key='name', name=replication_params['replication_name'])
    else:
        repl_session = self.get_replication_session(obj_nas, action=action)
        if repl_session and action and replication_params and (replication_params['replication_type'] == 'local') and (repl_session.remote_system.name != self.unity_conn.name):
            return None
    return repl_session
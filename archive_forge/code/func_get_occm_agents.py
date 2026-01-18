from __future__ import absolute_import, division, print_function
import uuid
import time
import base64
import json
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def get_occm_agents(self):
    if 'client_id' in self.parameters and self.parameters['state'] == 'absent':
        agent, error = self.na_helper.get_occm_agent_by_id(self.rest_api, self.parameters['client_id'])
        if error == '403' and b'Action not allowed for user' in agent:
            agents, error = ([], None)
            self.module.warn('Client Id %s was not found for this account.' % self.parameters['client_id'])
        else:
            agents = [agent]
    else:
        agents, error = self.na_helper.get_occm_agents_by_name(self.rest_api, self.parameters['account_id'], self.parameters['name'], 'GCP')
    if error:
        self.module.fail_json(msg='Error: getting OCCM agents: %s, %s' % (str(error), str(agents)))
    return agents
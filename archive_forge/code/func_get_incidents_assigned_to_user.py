from __future__ import absolute_import, division, print_function
from os import path
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils import deps
def get_incidents_assigned_to_user(self, pd_user_id):
    incident_info = {}
    incidents = self._apisession.list_all('incidents', params={'user_ids[]': [pd_user_id]})
    for incident in incidents:
        incident_info = {'title': incident['title'], 'key': incident['incident_key'], 'status': incident['status']}
    return incident_info
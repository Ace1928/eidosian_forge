from __future__ import absolute_import, division, print_function
from os import path
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils import deps
def add_pd_user(self, pd_name, pd_email, pd_role):
    try:
        user = self._apisession.persist('users', 'email', {'name': pd_name, 'email': pd_email, 'type': 'user', 'role': pd_role})
        return user
    except PDClientError as e:
        if e.response.status_code == 400:
            self._module.fail_json(msg='Failed to add %s due to invalid argument' % pd_name)
        if e.response.status_code == 401:
            self._module.fail_json(msg='Failed to add %s due to invalid API key' % pd_name)
        if e.response.status_code == 402:
            self._module.fail_json(msg='Failed to add %s due to inability to perform the action within the API token' % pd_name)
        if e.response.status_code == 403:
            self._module.fail_json(msg='Failed to add %s due to inability to review the requested resource within the API token' % pd_name)
        if e.response.status_code == 429:
            self._module.fail_json(msg='Failed to add %s due to reaching the limit of making requests' % pd_name)
from __future__ import (absolute_import, division, print_function)
import json
import zipfile
import io
def register_to_primary(self, primary):
    headers = {'Content-Type': 'application/json'}
    url = 'https://{primary_ip}/api/v1/deployment/node'.format(primary_ip=primary.ip)
    data = json.dumps({'fqdn': self.fqdn, 'userName': self.username, 'password': self.password, 'allowCertImport': True, 'roles': self.roles, 'services': self.services})
    try:
        response = requests.post(url=url, timeout=300, auth=(primary.username, primary.password), headers=headers, data=data, verify=False)
    except Exception as e:
        raise AnsibleActionFail(e)
    if not response:
        raise AnsibleActionFail('Failed to receive a valid response from the API. The actual response was: {response}'.format(response=response.text))
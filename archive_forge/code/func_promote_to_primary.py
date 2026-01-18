from __future__ import (absolute_import, division, print_function)
import json
import zipfile
import io
def promote_to_primary(self):
    headers = {'Content-Type': 'application/json'}
    url = 'https://{ip}/api/v1/deployment/primary'.format(ip=self.ip)
    try:
        response = requests.post(url=url, headers=headers, auth=(self.username, self.password), verify=False, timeout=60)
        if response.status_code == 200:
            return True
        else:
            raise AnsibleActionFail('Could not update node to PRIMARY')
    except Exception as e:
        raise AnsibleActionFail(e)
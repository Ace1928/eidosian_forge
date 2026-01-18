from __future__ import (absolute_import, division, print_function)
import json
import zipfile
import io
def return_id_of_certificate(self):
    url = 'https://{ip}/api/v1/certs/system-certificate/{hostname}'.format(ip=self.ip, hostname=self.hostname)
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    try:
        response = requests.get(url=url, timeout=15, headers=headers, auth=(self.username, self.password), verify=False)
    except requests.exceptions.ReadTimeout:
        raise AnsibleActionFail('The request timed out. Please verify that the API is enabled on the node.')
    except Exception as e:
        raise AnsibleActionFail(e)
    json_response = json.loads(response.text)
    for item in json_response.get('response'):
        if item.get('friendlyName') == 'Default self-signed server certificate':
            return item.get('id')
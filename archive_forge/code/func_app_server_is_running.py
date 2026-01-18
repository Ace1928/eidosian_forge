from __future__ import (absolute_import, division, print_function)
import json
import zipfile
import io
def app_server_is_running(self):
    url = 'https://{ip}/ers/config/op/systemconfig/iseversion'.format(ip=self.ip)
    headers = {'Accept': 'application/json'}
    try:
        response = requests.get(url=url, headers=headers, auth=(self.username, self.password), verify=False)
        if response.status_code == 502:
            return False
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        return False
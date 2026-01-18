from __future__ import absolute_import, division, print_function
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.module_utils.urls import open_url
import json
def call_url(self, url, url_username, url_password, url_path):
    """
        Execute the request against the API with the provided arguments and return json.
        """
    headers = {'Accept': 'application/json', 'X-HTTP-Method-Override': 'GET'}
    url = url + url_path
    rsp = open_url(url, url_username=self.url_username, url_password=self.url_password, force_basic_auth=self.force_basic_auth, headers=headers)
    content = ''
    if rsp:
        content = json.loads(rsp.read().decode('utf-8'))
        return content
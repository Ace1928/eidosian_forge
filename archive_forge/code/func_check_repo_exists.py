from __future__ import absolute_import, division, print_function
import json
import os
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.urls import url_argument_spec
def check_repo_exists(self, repo_id):
    try:
        self.get_repo_config_by_id(repo_id)
    except IndexError:
        return False
    else:
        return True
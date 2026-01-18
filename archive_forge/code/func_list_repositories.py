from __future__ import absolute_import, division, print_function
import os
from fnmatch import fnmatch
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
def list_repositories(self):
    """
        Generate RHSM repository list and return a list of dict
        """
    rc, out, err = self.run_repos(['--list'])
    repo_id = ''
    repo_name = ''
    repo_url = ''
    repo_enabled = ''
    repo_result = []
    for line in out.splitlines():
        if line == '' or line[0] == '+' or line[0] == ' ':
            continue
        if line.startswith('Repo ID: '):
            repo_id = line[9:].lstrip()
            continue
        if line.startswith('Repo Name: '):
            repo_name = line[11:].lstrip()
            continue
        if line.startswith('Repo URL: '):
            repo_url = line[10:].lstrip()
            continue
        if line.startswith('Enabled: '):
            repo_enabled = line[9:].lstrip()
            repo = {'id': repo_id, 'name': repo_name, 'url': repo_url, 'enabled': True if repo_enabled == '1' else False}
            repo_result.append(repo)
    return repo_result
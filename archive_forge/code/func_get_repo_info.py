from __future__ import annotations
import os
import re
from typing import Any
from streamlit import util
def get_repo_info(self):
    if not self.is_valid():
        return None
    remote_info = self.get_tracking_branch_remote()
    if remote_info is None:
        return None
    remote, branch = remote_info
    repo = None
    for url in remote.urls:
        https_matches = re.match(GITHUB_HTTP_URL, url)
        ssh_matches = re.match(GITHUB_SSH_URL, url)
        if https_matches is not None:
            repo = f'{https_matches.group(2)}/{https_matches.group(3)}'
            break
        if ssh_matches is not None:
            repo = f'{ssh_matches.group(1)}/{ssh_matches.group(2)}'
            break
    if repo is None:
        return None
    return (repo, branch, self.module)
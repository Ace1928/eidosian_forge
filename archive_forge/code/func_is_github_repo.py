from __future__ import annotations
import os
import re
from typing import Any
from streamlit import util
def is_github_repo(self) -> bool:
    if not self.is_valid():
        return False
    remote_info = self.get_tracking_branch_remote()
    if remote_info is None:
        return False
    remote, _branch = remote_info
    for url in remote.urls:
        if re.match(GITHUB_HTTP_URL, url) is not None or re.match(GITHUB_SSH_URL, url) is not None:
            return True
    return False
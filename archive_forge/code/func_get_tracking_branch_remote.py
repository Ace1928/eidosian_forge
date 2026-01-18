from __future__ import annotations
import os
import re
from typing import Any
from streamlit import util
def get_tracking_branch_remote(self):
    if not self.is_valid():
        return None
    tracking_branch = self.tracking_branch
    if tracking_branch is None:
        return None
    remote_name, *branch = tracking_branch.name.split('/')
    branch_name = '/'.join(branch)
    return (self.repo.remote(remote_name), branch_name)
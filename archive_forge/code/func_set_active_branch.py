from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def set_active_branch(self, branch_name: str) -> str:
    """Equivalent to `git checkout branch_name` for this Agent.
        Clones formatting from Github.

        Returns an Error (as a string) if branch doesn't exist.
        """
    curr_branches = [branch.name for branch in self.github_repo_instance.get_branches()]
    if branch_name in curr_branches:
        self.active_branch = branch_name
        return f'Switched to branch `{branch_name}`'
    else:
        return f'Error {branch_name} does not exist,in repo with current branches: {str(curr_branches)}'
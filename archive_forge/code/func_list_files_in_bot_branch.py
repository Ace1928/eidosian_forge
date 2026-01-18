from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def list_files_in_bot_branch(self) -> str:
    """
        Fetches all files in the active branch of the repo,
        the branch the bot uses to make changes.

        Returns:
            str: A plaintext list containing the the filepaths in the branch.
        """
    files: List[str] = []
    try:
        contents = self.github_repo_instance.get_contents('', ref=self.active_branch)
        for content in contents:
            if content.type == 'dir':
                files.extend(self.get_files_from_directory(content.path))
            else:
                files.append(content.path)
        if files:
            files_str = '\n'.join(files)
            return f'Found {len(files)} files in branch `{self.active_branch}`:\n{files_str}'
        else:
            return f'No files found in branch: `{self.active_branch}`'
    except Exception as e:
        return f'Error: {e}'
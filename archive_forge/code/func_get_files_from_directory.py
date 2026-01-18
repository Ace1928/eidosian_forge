from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def get_files_from_directory(self, directory_path: str) -> str:
    """
        Recursively fetches files from a directory in the repo.

        Parameters:
            directory_path (str): Path to the directory

        Returns:
            str: List of file paths, or an error message.
        """
    from github import GithubException
    files: List[str] = []
    try:
        contents = self.github_repo_instance.get_contents(directory_path, ref=self.active_branch)
    except GithubException as e:
        return f'Error: status code {e.status}, {e.message}'
    for content in contents:
        if content.type == 'dir':
            files.extend(self.get_files_from_directory(content.path))
        else:
            files.append(content.path)
    return str(files)
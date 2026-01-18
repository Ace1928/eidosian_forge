from __future__ import annotations
import urllib.parse
import warnings
from typing import Any
import deprecated
import urllib3
from urllib3 import Retry
import github
from github import Consts
from github.Auth import AppAuth
from github.GithubApp import GithubApp
from github.GithubException import GithubException
from github.Installation import Installation
from github.InstallationAuthorization import InstallationAuthorization
from github.PaginatedList import PaginatedList
from github.Requester import Requester
def get_repo_installation(self, owner: str, repo: str) -> Installation:
    """
        :calls: `GET /repos/{owner}/{repo}/installation <https://docs.github.com/en/rest/reference/apps#get-a-repository-installation-for-the-authenticated-app>`
        """
    owner = urllib.parse.quote(owner)
    repo = urllib.parse.quote(repo)
    return self._get_installed_app(url=f'/repos/{owner}/{repo}/installation')
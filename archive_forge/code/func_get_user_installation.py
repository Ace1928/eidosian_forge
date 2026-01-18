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
def get_user_installation(self, username: str) -> Installation:
    """
        :calls: `GET /users/{username}/installation <https://docs.github.com/en/rest/apps/apps#get-a-user-installation-for-the-authenticated-app>`
        """
    username = urllib.parse.quote(username)
    return self._get_installed_app(url=f'/users/{username}/installation')
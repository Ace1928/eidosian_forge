from __future__ import annotations
import urllib.parse
from typing import TYPE_CHECKING, Any
import github.AccessToken
import github.Auth
from github.GithubException import BadCredentialsException, GithubException
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
from github.Requester import Requester
@property
def client_secret(self) -> str:
    return self._client_secret.value
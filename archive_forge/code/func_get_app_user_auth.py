from __future__ import annotations
import urllib.parse
from typing import TYPE_CHECKING, Any
import github.AccessToken
import github.Auth
from github.GithubException import BadCredentialsException, GithubException
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
from github.Requester import Requester
def get_app_user_auth(self, token: AccessToken) -> AppUserAuth:
    return github.Auth.AppUserAuth(client_id=self.client_id, client_secret=self.client_secret, token=token.token, token_type=token.type, expires_at=token.expires_at, refresh_token=token.refresh_token, refresh_expires_at=token.refresh_expires_at, requester=self._requester)
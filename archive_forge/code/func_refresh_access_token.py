from __future__ import annotations
import urllib.parse
from typing import TYPE_CHECKING, Any
import github.AccessToken
import github.Auth
from github.GithubException import BadCredentialsException, GithubException
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
from github.Requester import Requester
def refresh_access_token(self, refresh_token: str) -> AccessToken:
    """
        :calls: `POST /login/oauth/access_token <https://docs.github.com/en/developers/apps/identifying-and-authorizing-users-for-github-apps>`_
        :param refresh_token: string
        """
    assert isinstance(refresh_token, str)
    post_parameters = {'client_id': self.client_id, 'client_secret': self.client_secret, 'grant_type': 'refresh_token', 'refresh_token': refresh_token}
    headers, data = self._checkError(*self._requester.requestJsonAndCheck('POST', 'https://github.com/login/oauth/access_token', headers={'Accept': 'application/json'}, input=post_parameters))
    return github.AccessToken.AccessToken(requester=self._requester, headers=headers, attributes=data, completed=False)
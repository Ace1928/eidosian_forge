from __future__ import annotations
import urllib.parse
from typing import TYPE_CHECKING, Any
import github.AccessToken
import github.Auth
from github.GithubException import BadCredentialsException, GithubException
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
from github.Requester import Requester
def get_access_token(self, code: str, state: str | None=None) -> AccessToken:
    """
        :calls: `POST /login/oauth/access_token <https://docs.github.com/en/developers/apps/identifying-and-authorizing-users-for-github-apps>`_
        """
    assert isinstance(code, str), code
    post_parameters = {'code': code, 'client_id': self.client_id, 'client_secret': self.client_secret}
    if state is not None:
        post_parameters['state'] = state
    headers, data = self._checkError(*self._requester.requestJsonAndCheck('POST', 'https://github.com/login/oauth/access_token', headers={'Accept': 'application/json'}, input=post_parameters))
    return github.AccessToken.AccessToken(requester=self._requester, headers=headers, attributes=data, completed=False)
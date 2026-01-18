from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def github_com_two_factor_auth(self) -> bool:
    self._completeIfNotSet(self._github_com_two_factor_auth)
    return self._github_com_two_factor_auth.value
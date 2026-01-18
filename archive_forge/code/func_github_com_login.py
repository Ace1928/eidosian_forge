from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def github_com_login(self) -> str:
    self._completeIfNotSet(self._github_com_login)
    return self._github_com_login.value
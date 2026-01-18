from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def github_com_enterprise_roles(self) -> list:
    self._completeIfNotSet(self._github_com_enterprise_roles)
    return self._github_com_enterprise_roles.value
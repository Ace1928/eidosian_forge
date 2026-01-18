from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def github_com_profile(self) -> str:
    self._completeIfNotSet(self._github_com_profile)
    return self._github_com_profile.value
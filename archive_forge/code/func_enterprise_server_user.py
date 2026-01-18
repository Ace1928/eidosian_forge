from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def enterprise_server_user(self) -> bool:
    self._completeIfNotSet(self._enterprise_server_user)
    return self._enterprise_server_user.value
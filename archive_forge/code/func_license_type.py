from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def license_type(self) -> str:
    self._completeIfNotSet(self._license_type)
    return self._license_type.value
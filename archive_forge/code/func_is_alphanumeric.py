from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def is_alphanumeric(self) -> bool:
    return self._is_alphanumeric.value
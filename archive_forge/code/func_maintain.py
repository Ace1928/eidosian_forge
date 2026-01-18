from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def maintain(self) -> bool:
    return self._maintain.value
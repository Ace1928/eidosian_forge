from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def deletions(self) -> int:
    return self._deletions.value
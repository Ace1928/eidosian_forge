from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def annotations_count(self) -> int:
    return self._annotations_count.value
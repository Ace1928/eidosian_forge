from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def annotations_url(self) -> str:
    return self._annotations_url.value
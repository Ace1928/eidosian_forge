from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def raw_url(self) -> str:
    return self._raw_url.value
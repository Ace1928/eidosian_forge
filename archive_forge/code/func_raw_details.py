from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def raw_details(self) -> str:
    return self._raw_details.value
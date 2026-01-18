from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def cwe_id(self) -> str:
    return self._cwe_id.value
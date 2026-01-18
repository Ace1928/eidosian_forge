from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def blob_url(self) -> str:
    return self._blob_url.value
from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def latest_comment_url(self) -> str:
    return self._latest_comment_url.value
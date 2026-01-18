from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def previous_filename(self) -> str:
    return self._previous_filename.value
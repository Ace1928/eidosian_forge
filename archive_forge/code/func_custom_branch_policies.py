from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def custom_branch_policies(self) -> bool:
    return self._custom_branch_policies.value
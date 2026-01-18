from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def github_com_orgs_with_pending_invites(self) -> list:
    self._completeIfNotSet(self._github_com_orgs_with_pending_invites)
    return self._github_com_orgs_with_pending_invites.value
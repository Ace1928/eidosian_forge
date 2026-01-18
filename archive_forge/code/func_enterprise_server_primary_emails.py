from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def enterprise_server_primary_emails(self) -> list:
    self._completeIfNotSet(self._enterprise_server_primary_emails)
    return self._enterprise_server_primary_emails.value
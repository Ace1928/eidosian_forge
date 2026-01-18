from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def github_com_saml_name_id(self) -> str:
    self._completeIfNotSet(self._github_com_saml_name_id)
    return self._github_com_saml_name_id.value